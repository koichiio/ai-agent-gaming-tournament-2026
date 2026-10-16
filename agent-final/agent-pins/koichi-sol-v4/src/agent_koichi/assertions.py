"""Bounded assertion/hypothesis scores for already validated dialogue claims."""

import math
from dataclasses import dataclass

ROLE_WEIGHT = 0.20
LEGITIMACY_WEIGHT = 0.15
CHECK_TRUTH_WEIGHT = 0.25
CREDIBILITY = {"seer": 0.80, "villager": 0.60, "wolf": 0.25}

type World = tuple[tuple[int, int], int]


@dataclass(frozen=True)
class Hypothesis:
    """Content truth and speaker legitimacy in one world, not hard evidence."""

    truth: bool
    legitimate: bool


@dataclass(frozen=True)
class Assertion:
    """Minimal proposition compiled only after BeliefEngine's evidence filters."""

    speaker: int
    kind: str
    target: int
    value: str

    @classmethod
    def from_event(cls, event: dict) -> Assertion:
        if event["kind"] == "role_claim":
            return cls(event["speaker"], "role_claim", event["speaker"], event["role"])
        return cls(event["speaker"], "check_claim", event["target"], event["result"])

    def evaluate(self, world: World) -> Hypothesis:
        wolves, seer = world
        if self.kind == "role_claim":
            role = (
                "wolf"
                if self.speaker in wolves
                else "seer"
                if self.speaker == seer
                else "villager"
            )
            return Hypothesis(role == self.value, True)
        return Hypothesis(
            (self.target in wolves) == (self.value == "wolf"), self.speaker == seer
        )


def score_world(world: World, assertions: dict[int, list[Assertion]]) -> float:
    """Marginalize faithful vs uninformative speaker accounts in log space.

    Each speaker contributes at most +/-0.60 log units, regardless of history
    length. The caller owns validation, deduplication, and hard conditioning.
    """
    wolves, seer = world
    scores = []
    for speaker, statements in assertions.items():
        role_terms, check_terms = [], []
        for assertion in statements:
            hypothesis = assertion.evaluate(world)
            if assertion.kind == "role_claim":
                role_terms.append(ROLE_WEIGHT * (2 * hypothesis.truth - 1))
            else:
                check_terms.append(
                    LEGITIMACY_WEIGHT * (2 * hypothesis.legitimate - 1)
                    + CHECK_TRUTH_WEIGHT
                    * hypothesis.legitimate
                    * (2 * hypothesis.truth - 1)
                )
        energy = math.fsum(
            math.fsum(terms) / len(terms)
            for terms in (role_terms, check_terms)
            if terms
        )
        role = (
            "wolf" if speaker in wolves else "seer" if speaker == seer else "villager"
        )
        credibility = CREDIBILITY[role]
        # log((1-c) + c*exp(energy)), stable even when energy is close to zero.
        scores.append(math.log1p(credibility * math.expm1(energy)))
    return math.fsum(scores)
