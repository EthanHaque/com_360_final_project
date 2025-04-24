{
  inputs = {
    nixpkgs.url = "github:cachix/devenv-nixpkgs/rolling";
    systems.url  = "github:nix-systems/default";
    devenv.url   = "github:cachix/devenv";
    devenv.inputs.nixpkgs.follows = "nixpkgs";
    nixpkgs-python.url = "github:cachix/nixpkgs-python";
  };

  nixConfig = {
    extra-trusted-public-keys = "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=";
    extra-substituters = "https://devenv.cachix.org";
  };

  outputs = { self, nixpkgs, devenv, systems, ... }@inputs:
    let
      forEachSystem = nixpkgs.lib.genAttrs (import systems);
    in {

      packages = forEachSystem (system: {
        devenv-up   = self.devShells.${system}.default.config.procfileScript;
        devenv-test = self.devShells.${system}.default.config.test;
      });

      devShells = forEachSystem (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};

          pythonDeps = pkgs.python311.withPackages (ps: with ps; [
            matplotlib
            numpy
            pandas
            structlog
            polars
            pyarrow
          ]);
        in {
          default = devenv.lib.mkShell {
            inherit inputs pkgs;
            modules = [
              {
                packages = with pkgs; [
                  pre-commit
                  pythonDeps
                ];

                languages.python = {
                  enable  = true;
                  uv.enable = true;
                  venv = {
                    enable = true;
                    quiet  = true;
                  };
                };
              }
            ];
          };
        });
    };
}
