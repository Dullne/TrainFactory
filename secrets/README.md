# Runtime secrets

Do not place secret values in this directory. It contains instructions only;
generated verification credentials belong under the ignored `.runtime`
directory. Never commit a username, email address, password, token, secret URL,
or machine-specific absolute path.

## Choose exactly one mode

**Direct mode** is the local compatibility path. Set `MYSQL_ROOT_PASSWORD`,
`MYSQL_APP_PASSWORD`, `JWT_SECRET_KEY`, and `DEFAULT_ADMIN_PASSWORD` in the
operator-owned `.env`, and leave every `*_SECRET_PATH` empty.

**File mode** is an explicit Compose secrets opt-in. Leave the corresponding
direct values empty, provision the five secret files outside Git, set the five
`*_SECRET_PATH` variables in an operator-owned env file, and add
`docker/docker-compose.secrets.yml` after the base Compose file. A direct value
and its file source must never be set together; the entrypoint and Settings
reject mixed or empty file sources before the API starts.

Always validate the selected shape before starting containers:

```powershell
$secretEnv = '<operator-owned-path-env>'
$composeArgs = @(
  '--env-file', '.env', '--env-file', $secretEnv,
  '-f', 'docker/docker-compose.yml',
  '-f', 'docker/docker-compose.secrets.yml'
)
docker compose @composeArgs config --quiet
docker compose @composeArgs up -d --wait --wait-timeout 600 `
  mysql train-factory-api train-factory-web
```

## Isolated verification bundles

The materializer is for short-lived `verify` and `ci` runs, not for production
secret custody. From the repository root in PowerShell, create a verification
bundle with a unique 32-character lowercase hexadecimal run ID:

```powershell
$runId = [guid]::NewGuid().ToString("N")
python -I scripts/materialize_compose_secrets.py create `
  --scope verify `
  --run-id $runId `
  --output-root .runtime `
  --env-out ".runtime/verify-$runId/compose-secrets.env" `
  --state-out ".runtime/verify-$runId/compose-secrets.state.json"
```

Use the generated env file only for the matching Compose invocation. The
materializer restricts the directory and files to the current user (POSIX
`0700`/`0600`, or a non-inherited Windows ACL).

Always remove the bundle from a `finally` block or an `if: always()` CI step:

```powershell
python -I scripts/materialize_compose_secrets.py cleanup `
  --state ".runtime/verify-$runId/compose-secrets.state.json"
```
