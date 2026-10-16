{
  description = "Development shell for the AI agent tournament";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      systems,
      treefmt-nix,
    }:
    let
      eachSystem = f: nixpkgs.lib.genAttrs (import systems) (system: f nixpkgs.legacyPackages.${system});
      formatting = eachSystem (pkgs: treefmt-nix.lib.evalModule pkgs ./treefmt.nix);
    in
    {
      devShells = eachSystem (pkgs: {
        default = pkgs.mkShellNoCC {
          packages = [
            pkgs.bashInteractive
            pkgs.git
            pkgs.python3
            pkgs.uv
            formatting.${pkgs.stdenv.hostPlatform.system}.config.build.wrapper
          ];

          shellHook = ''
            # Append the hook setting without replacing inherited Git settings.
            hookIndex=''${GIT_CONFIG_COUNT:-0}
            export "GIT_CONFIG_KEY_$hookIndex=core.hooksPath"
            export "GIT_CONFIG_VALUE_$hookIndex=$(git rev-parse --show-toplevel)/.githooks"
            export GIT_CONFIG_COUNT=$((hookIndex + 1))
            unset hookIndex
          '';
        };
      });

      formatter = eachSystem (pkgs: formatting.${pkgs.stdenv.hostPlatform.system}.config.build.wrapper);
      checks = eachSystem (pkgs: {
        formatting = formatting.${pkgs.stdenv.hostPlatform.system}.config.build.check self;
      });
    };
}
