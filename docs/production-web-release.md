# Production Web-only Release Runbook

Use this procedure only when the production API image and API revision must stay
unchanged and a previously built immutable Web image is being promoted. Use the
full release procedure when the API changes, a migration is required, or the
current production manifest is not valid.

## Preconditions

- Run from the trusted production checkout with Docker Engine available.
- `.runtime` must be a private directory owned by the release account.
- `.runtime/release.env` and `.runtime/production-compose-manifest.json` must
  describe the currently running production release.
- The requested Web image must exist locally by the supplied immutable
  `sha256:` image ID and carry the expected OCI revision label.
- The production API, database, fixed containers, network, and volumes must be
  healthy and unchanged.
- Run only one release operation at a time. The process lock serializes
  cooperating invocations. On POSIX, other processes running as the same UID
  are inside the trusted release boundary and must not modify `.runtime`.

Resolve the intended immutable identity, then run the isolated entry point:

```powershell
$webImage = '<registry>/<web-image>:<version>'
$webImageId = docker image inspect --format '{{.Id}}' $webImage
$webRevision = '<40-character-lowercase-git-sha>'

python -I scripts/promote_web_release.py `
  --web-image $webImage `
  --web-image-id $webImageId `
  --web-revision $webRevision
```

Do not invoke `docker compose up` manually with `.runtime/release.env`. The
promotion entry point freezes a new manifest, executes the validated Compose
JSON with API and Web images pinned by immutable IDs, and permits only the
exact `train-factory-web` replacement command.

## Success Contract

A zero exit code and JSON result mean that all of these checks passed:

- the API container ID and API image ID stayed unchanged;
- fixed containers, network endpoints, and volumes stayed unchanged;
- the new Web container is healthy and uses the requested image ID;
- `/login` and `/training` return the SPA shell through the candidate Web;
- API health succeeds directly, and public auth configuration matches through
  the Web proxy; and
- an unprivileged invalid auth cookie is rejected with the expected 401
  contract.

The HTTP probe does not execute browser JavaScript, perform a valid login, or
prove a browser password-save prompt. Validate rendered DOM behavior and the
password manager interaction separately in a real supported browser when that
is part of the release acceptance criteria.

## Failure And Recovery

When the command says `recovery is required before retry`, preserve all files
in `.runtime`, inspect `web-promotion-active.json`, `.claim`, `.retiring`,
`.tmp`, and `.delete` entries, and confirm that no external resource or file
has drifted. Then rerun the exact same command. The entry point automatically
recovers the recorded transaction before attempting the requested promotion.
Do not delete or edit transaction files to force a retry.

When the command says `temporary state requires inspection`, `manual
inspection is required`, or `terminal state requires manual inspection`, stop
and escalate. Do not mutate transaction files or retry blindly; these outcomes
are intentionally outside automatic recovery.

When the command reports that the old Web was restored, the requested release
was rejected and the previous Web image is running. Re-run only after correcting
the original validation or runtime failure.
