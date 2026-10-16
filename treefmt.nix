{
  projectRootFile = "flake.nix";

  programs = {
    mdformat = {
      enable = true;
      settings = {
        wrap = 80;
        number = false;
      };
      plugins = ps: [
        ps.mdformat-frontmatter
        ps.mdformat-gfm
        ps.mdformat-myst
      ];
    };
    nixfmt.enable = true;
    ruff-format.enable = true;
    shfmt.enable = true;
  };

  settings.formatter.shfmt.includes = [ ".githooks/*" ];

  # Match records are append-only data, not formatter inputs.
  settings.global.excludes = [
    "**/data/results.jsonl"
    "**/data/logs/**"
  ];
}
