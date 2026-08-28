# Contributing

Thank you for contributing to `lithos-llm`.

## Before you start

Open an issue before a large change or a change to the public API. For security
issues, follow [SECURITY.md](SECURITY.md) instead of opening a public issue.

## Set up the repository

Install [Mise](https://mise.jdx.dev/), then run:

```sh
mise trust
mise install --locked
mise run setup
```

The setup task prepares the pinned Rust style guide used by this repository.

## Make a change

- Keep public provider-neutral types separate from private codecs and
  transport.
- Treat serialized request, response, content, and catalog shapes as public
  contracts.
- Add tests for behavior that changes.
- Update `CHANGELOG.md` for a user-visible change.
- Do not include credentials, provider response bodies, or other secrets in
  tests, logs, issues, or pull requests.

## Verify the change

Run the routine verification gate:

```sh
mise run check
```

Run the extended gate for dependency, feature, MSRV, or release changes:

```sh
mise run check:nightly
```

## Submit the change

Use a focused commit with a direct commit message. In the pull request, explain
the behavior change, compatibility impact, and verification you ran.
