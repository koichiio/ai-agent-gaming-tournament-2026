"""Match-local claims with authenticated envelopes and explicit source links."""

import math

from .belief import MIN_CLAIM_CONFIDENCE, EvidenceError

_KINDS = ("role_claim", "check_claim", "accuse", "defend", "vote_intent", "retract")
_ROLES = ("wolf", "seer", "villager")
_FIELDS = {
    "source_id",
    "kind",
    "target",
    "role",
    "result",
    "check_day",
    "quote",
    "confidence",
    "uncertain",
    "negated",
    "quoted",
}


def authorized_message(message: dict, self_id: int) -> dict:
    """Validate visibility before any message reaches a model or audit trace."""
    if not isinstance(message, dict):
        raise EvidenceError("Message envelope must be a dictionary.")
    sender, recipients = message.get("sender"), message.get("recipients")
    visibility = message.get("visibility")
    if (
        type(sender) is not int
        or not 0 <= sender < 7
        or not isinstance(message.get("content"), str)
        or not isinstance(recipients, list)
        or len(recipients) > 1
        or any(type(seat) is not int or not 0 <= seat < 7 for seat in recipients)
        or sender in recipients
        or message.get("type") not in ("chat", "terminate")
        or visibility not in ("broadcast", "direct")
        or (visibility == "broadcast" and recipients)
        or (
            visibility == "direct"
            and (
                len(recipients) != 1
                or (sender != self_id and self_id not in recipients)
            )
        )
    ):
        raise EvidenceError("Malformed or unauthorized authenticated message envelope.")
    return {
        key: value
        for key, value in message.items()
        if key
        in (
            "message_id",
            "seq",
            "sender",
            "recipients",
            "visibility",
            "type",
            "content",
            "state_version",
            "sent_at",
        )
    }


class ClaimLedger:
    """Single-writer ledger; confidence measures extraction, not truthfulness.

    Exact repeated claims share one event, with additional provenance retained
    in ``repetitions``. Belief scoring selects an eligible occurrence using its
    own confidence and provenance, without changing the canonical source or
    retraction identity.
    Retractions must select one prior same-speaker event;
    quoted/negated/uncertain or low-confidence retractions cannot deactivate it.
    The supplied day records observation time, not an inferred message date.
    """

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.processed: set[str] = set()
        self.rejected: list[dict] = []
        self._self_id: int | None = None
        self._sources: dict[str, dict] = {}
        self._references: dict[str, list[dict]] = {}
        self._by_key: dict[tuple, dict] = {}

    @staticmethod
    def source_id(message: dict) -> str:
        if not isinstance(message, dict):
            raise EvidenceError("Message envelope must be a dictionary.")
        if "message_id" in message:
            message_id = message["message_id"]
            if not isinstance(message_id, str) or not message_id.strip():
                raise EvidenceError("Message ID must be a nonempty string.")
            return message_id
        seq = message.get("seq")
        if type(seq) is not int or seq < 0:
            raise EvidenceError(
                "Message without an ID requires a nonnegative sequence."
            )
        return f"seq:{seq}"

    def add_extraction(
        self, messages: list[dict], extraction: dict, self_id: int, day: int
    ) -> list[dict]:
        """Accept supported events, recording individual rejects without raw text.

        A malformed extraction/envelope raises before processing any sources.
        A valid extraction (including an empty list) consumes its supplied sources;
        invalid individual events go to ``rejected`` and never become facts.
        """
        if (
            not isinstance(extraction, dict)
            or set(extraction) != {"events"}
            or not isinstance(extraction["events"], list)
        ):
            raise EvidenceError(
                "Extraction must be a dictionary containing an events list."
            )
        if (
            not isinstance(messages, list)
            or type(self_id) is not int
            or not 0 <= self_id < 7
            or type(day) is not int
            or day < 1
            or (self._self_id is not None and self._self_id != self_id)
        ):
            raise EvidenceError(
                "Invalid message batch, observer seat, or observation day."
            )

        sources: dict[str, dict] = {}
        for message in messages:
            source_id = self.source_id(message)
            sender, recipients = message.get("sender"), message.get("recipients")
            visibility, message_type = message.get("visibility"), message.get("type")
            if (
                type(sender) is not int
                or not 0 <= sender < 7
                or not isinstance(message.get("content"), str)
                or not isinstance(recipients, list)
                or len(recipients) > 1
                or any(
                    type(seat) is not int or not 0 <= seat < 7 for seat in recipients
                )
                or sender in recipients
                or visibility not in ("broadcast", "direct")
                or message_type not in ("chat", "terminate")
                or (visibility == "broadcast" and recipients)
                or (visibility == "direct" and len(recipients) != 1)
                or (
                    visibility == "direct"
                    and sender != self_id
                    and self_id not in recipients
                )
            ):
                raise EvidenceError(
                    "Malformed or unauthorized authenticated message envelope."
                )
            source = {
                "source_id": source_id,
                "speaker": sender,
                "source_text": message["content"],
                "visibility": visibility,
                "recipients": list(recipients),
                "message_type": message_type,
            }
            for field in ("seq", "state_version", "sent_at"):
                if field not in message:
                    continue
                value = message[field]
                if (
                    field in ("seq", "state_version")
                    and (type(value) is not int or value < 0)
                ) or (field == "sent_at" and not isinstance(value, str)):
                    raise EvidenceError("Malformed authenticated message provenance.")
                source[field] = value
            previous = sources.get(source_id, self._sources.get(source_id))
            if previous is not None and previous != source:
                raise EvidenceError(
                    "The same authenticated source ID has conflicting envelopes."
                )
            sources[source_id] = source

        pending: list[tuple[int, dict]] = []

        def reject(index: int, event: object, reason: str) -> None:
            source_id = event.get("source_id") if isinstance(event, dict) else None
            self.rejected.append(
                {
                    "event_index": index,
                    "source_id": (
                        source_id
                        if isinstance(source_id, str) and source_id in sources
                        else None
                    ),
                    "reason": reason,
                }
            )

        for index, event in enumerate(extraction["events"]):
            if (
                not isinstance(event, dict)
                or not _FIELDS.issubset(event)
                or set(event) - (_FIELDS | {"retracts"})
            ):
                reject(
                    index, event, "Missing required fields or unsupported event fields."
                )
                continue
            source_id = event["source_id"]
            if not isinstance(source_id, str) or source_id not in sources:
                reject(
                    index, event, "Unknown source ID; no speaker override is permitted."
                )
                continue
            if source_id in self.processed:
                continue
            source = sources[source_id]
            kind, target = event["kind"], event["target"]
            confidence, check_day = event["confidence"], event["check_day"]
            if (
                kind not in _KINDS
                or (
                    target is not None
                    and (type(target) is not int or not 0 <= target < 7)
                )
                or event["role"] not in (None, *_ROLES)
                or event["result"] not in (None, "wolf", "villager")
                or (
                    check_day is not None
                    and (type(check_day) is not int or check_day < 1)
                )
                or type(confidence) not in (int, float)
                or not 0 <= confidence <= 1
                or not math.isfinite(confidence)
                or any(
                    type(event[flag]) is not bool
                    for flag in ("uncertain", "negated", "quoted")
                )
                or not isinstance(event["quote"], str)
                or not event["quote"].strip()
                or event["quote"] not in source["source_text"]
                or source["message_type"] != "chat"
            ):
                reject(
                    index,
                    event,
                    "Invalid types or values, or an unauthenticated quote.",
                )
                continue
            retracts = event.get("retracts")
            if (
                (
                    kind == "role_claim"
                    and (
                        event["role"] is None
                        or event["result"] is not None
                        or check_day is not None
                    )
                )
                or (
                    kind == "check_claim"
                    and (
                        target is None
                        or event["result"] is None
                        or event["role"] not in (None, "seer")
                    )
                )
                or (kind in ("accuse", "defend") and target is None)
                or (kind not in ("check_claim", "retract") and check_day is not None)
                or (
                    kind == "vote_intent"
                    and (event["role"] is not None or event["result"] is not None)
                )
                or (kind != "retract" and retracts is not None)
                or (
                    kind == "retract"
                    and (
                        not isinstance(retracts, str)
                        or not retracts
                        or retracts == source_id
                    )
                )
            ):
                reject(
                    index,
                    event,
                    "Fields do not identify a supported claim or explicit retraction.",
                )
                continue
            accepted = {**event, **source, "day": day}
            if kind == "role_claim" and target is None:
                accepted["target"] = source["speaker"]
            accepted["active"] = source["speaker"] != self_id and not any(
                accepted[flag] for flag in ("uncertain", "negated", "quoted")
            )
            pending.append((index, accepted))

        # Envelope sequence, not the extractor's array order, orders new sources.
        ordered_sources = list(sources)
        if all("seq" in source for source in sources.values()):
            ordered_sources.sort(key=lambda source_id: sources[source_id]["seq"])
        order = {source_id: index for index, source_id in enumerate(ordered_sources)}
        pending.sort(key=lambda item: order[item[1]["source_id"]])
        self._sources.update(sources)
        self._self_id = self_id
        newly_accepted: list[dict] = []
        for index, event in pending:
            source_id, kind = event["source_id"], event["kind"]
            prior = None
            if kind == "retract":
                candidates = [
                    claim
                    for claim in self._references.get(event["retracts"], [])
                    if claim["kind"] != "retract"
                    and claim["speaker"] == event["speaker"]
                    and all(
                        event[field] is None or event[field] == claim[field]
                        for field in ("target", "result", "check_day")
                    )
                    and (
                        event["role"] is None
                        or event["role"] == claim["role"]
                        or (claim["kind"] == "check_claim" and event["role"] == "seer")
                    )
                ]
                if len(candidates) != 1:
                    reject(
                        index,
                        event,
                        "Retraction must select one prior same-speaker event.",
                    )
                    continue
                prior = candidates[0]
                linked_source = self._sources[event["retracts"]]
                if (
                    "seq" in event
                    and "seq" in linked_source
                    and event["seq"] <= linked_source["seq"]
                ):
                    reject(
                        index,
                        event,
                        "Retraction cannot reference a simultaneous or future source.",
                    )
                    continue
                if event["confidence"] < MIN_CLAIM_CONFIDENCE:
                    event["active"] = False

            if (
                prior is not None
                and event["confidence"] >= MIN_CLAIM_CONFIDENCE
                and not any(event[flag] for flag in ("uncertain", "negated", "quoted"))
            ):
                prior["active"] = False
                retractions = prior.setdefault("retracted_by", [])
                if source_id not in retractions:
                    retractions.append(source_id)
                event["retracted_source_id"] = prior["source_id"]

            key = tuple(
                None if field == "role" and kind == "check_claim" else event[field]
                for field in (
                    "speaker",
                    "kind",
                    "target",
                    "role",
                    "result",
                    "check_day",
                    "uncertain",
                    "negated",
                    "quoted",
                )
            ) + (event.get("retracts"),)
            previous = self._by_key.get(key)
            if previous is not None and "retracted_by" not in previous:
                # Preserve distinct parsing confidence even within one source.
                if (
                    source_id != previous["source_id"]
                    or event["confidence"] != previous["confidence"]
                ) and not any(
                    repeat["source_id"] == source_id
                    and repeat["confidence"] == event["confidence"]
                    for repeat in previous.get("repetitions", [])
                ):
                    previous.setdefault("repetitions", []).append(dict(event))
                references = self._references.setdefault(source_id, [])
                if not any(reference is previous for reference in references):
                    references.append(previous)
                continue

            self.events.append(event)
            self._by_key[key] = event
            self._references.setdefault(source_id, []).append(event)
            newly_accepted.append(event)
        self.processed.update(sources)
        return newly_accepted
