import argparse


def parse_toy_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--profile", required=True)
    return parser.parse_args(argv)


def provider_available(name: str, providers: list[str]) -> bool:
    return name in providers
