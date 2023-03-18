{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  };

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
  in
    {
      packages.${system}.default = pkgs.poetry2nix.mkPoetryApplication {
        projectDir = self;
      };

      devShells.${system}.default = pkgs.mkShellNoCC {
        packages = with pkgs; [
          (poetry2nix.mkPoetryEnv { projectDir = self; })
          pyright
        ];
      };
    };
}
