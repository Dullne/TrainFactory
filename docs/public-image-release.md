# Public Image Release

## Docker Hub publication

The current release destination is Docker Hub. Publish the project-owned
`<dockerhub-namespace>/trainfactory-api` and
`<dockerhub-namespace>/trainfactory-web` repositories only. The namespace is
the operator's verified Docker Hub account; do not embed registry credentials
in source, build arguments, or release notes.

Use an independent public GitHub clone on `master`, review the complete range
from the previous public revision, and require all eight CI jobs to pass for
the exact commit. Private checkout history, internal configuration, and local
data must never enter the public build context. Build the clean public clone
with `python -I scripts/build_release.py --build` using an empty temporary
Docker client configuration. Keep the pinned GPU providers unchanged.

After preparation, use native Docker commands with the operator's credential
store to tag and push the validated immutable image IDs. Publish only
`<version>-<short-commit-sha>` and `sha-<full-commit-sha>`; do not overwrite a
different existing image or publish a floating `latest` tag. Verify both remote
manifest config digests against the local image IDs, and verify `linux/amd64`
and the OCI source/revision/version labels. Record the immutable remote
digests in the release notes. Publishing does not restart running services;
deployment is a separate operation.

## Legacy GHCR preparation utility

`scripts/prepare_public_release.py` remains a GHCR-only preparation utility.
It does not publish to Docker Hub and should not be used to select a Docker
Hub namespace. Its preparation and authentication boundaries below also apply
to the Docker Hub workflow above.

The legacy utility prepares only the two project-owned runtime images:

- `ghcr.io/dullne/trainfactory-api`
- `ghcr.io/dullne/trainfactory-web`

The digest-pinned vLLM, SGLang, Xinference, Milvus, MySQL, and other upstream
images in `docker/images.lock.env` are pulled from their upstream registries;
they are not republished as TrainFactory images.

## Security boundary

The release is deliberately split into two phases:

1. **Registry-unauthenticated preparation** validates a fresh public GitHub clone, the
   exact public `master` revision, the complete successful CI run, tracked
   files, and the locally built images. It uses an empty temporary Docker
   client configuration and writes a publish plan without registry credentials.
2. **Authenticated publication** is an operator action. After logging in, run
   only reviewed, fixed native Docker `tag`, `push`, and registry inspection
   commands. Do not run repository Python, npm, build, or shell code while the
   registry credential is available.

Never prepare a public image from a private or internal checkout or its history.
The preparation command accepts no repository, registry, image, tag, token, or
force override.

The preparation script is still local Python code with the operating system
permissions of its caller. Run it in a dedicated account or environment that
does not contain private source, SSH keys, registry credentials, or unrelated
secrets. The tracked-secret scanner covers a small fixed set of credential
patterns; it does not replace a complete review of the commits and files that
will become public.

## Phase 1: prepare without credentials

Use a new full clone with the one exact HTTPS remote. Do not reuse a worktree,
shared object store, shallow clone, or checkout with extra remotes.
Python 3.11 or newer is required for this release utility.

```powershell
git clone https://github.com/Dullne/TrainFactory.git TrainFactory-public-release
Set-Location TrainFactory-public-release
git switch master
git pull --ff-only origin master
python -I scripts/prepare_public_release.py --prepare
```

The command fails before Docker build unless all of these conditions hold:

- `HEAD`, local `master`, `origin/master`, and the GitHub REST master ref are
  the same 40-character commit;
- the checkout is clean, standalone, full, and contains no tracked symlink,
  gitlink, replacement ref, special index flag, unsafe Git rewrite, extra
  remote, or push URL;
- the exact `CI` workflow run for that commit is a completed successful
  `master` push and all eight expected jobs completed successfully;
- the tracked secret scan succeeds;
- both local images are `linux/amd64`, have the expected OCI identity labels,
  and have non-zero immutable image IDs.

The Docker build never uses the operator checkout as its context. The script
creates another temporary, full, non-local clone from the fixed public HTTPS
URL, verifies that clone and its complete file set, and builds only from it.

On success it writes the ignored local artifact
`.runtime/ghcr-publish-plan.json`. Starting another preparation attempt first
invalidates an older plan and takes an exclusive local preparation lock.

The plan contains only the two fixed GHCR repositories, local image IDs,
expected labels, CI evidence, and these non-floating tags:

- `sha-<full-commit-sha>`
- `<version>-<short-commit-sha>`

It never creates `latest`, logs in to a registry, or pushes an image.

## Phase 2: publish with a short-lived credential

Before login, review the plan and copy its fixed native Docker operations into
an operator-controlled location outside the clone. For a personal GHCR
account, create a short-lived classic personal access token with only
`write:packages`. Do not grant `repo` or `delete:packages` for this operation.
Create a dedicated temporary Docker client configuration so the login neither
uses nor overwrites the normal Docker credential store. Pass the token through
standard input, remove it from the environment, and do not print it:

```powershell
$publishDockerConfig = Join-Path ([IO.Path]::GetTempPath()) `
  ("trainfactory-ghcr-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $publishDockerConfig | Out-Null
$env:DOCKER_CONFIG = $publishDockerConfig
$env:CR_PAT | docker login ghcr.io -u <github-user> --password-stdin
Remove-Item Env:CR_PAT
```

Immediately before tagging, inspect each planned `sha256:<local-image-id>` and
require its ID, `linux/amd64` platform, and OCI labels to match the plan. Use
that image ID—not the mutable local image tag—as the source of every fixed
GHCR tag. Refuse to overwrite a remote tag that already exists with different
content.

Publish exactly the API and Web tags recorded in the plan. Never use
`--all-tags` or add `latest`. After each push, resolve its remote manifest
digest and inspect the immutable `<repository>@sha256:<manifest-digest>`
reference. The remote manifest config digest must equal the planned local image
ID, and the platform and OCI labels must match the plan. Only after both images
pass remote verification may their digest references be used for deployment.

Finally log out, delete the temporary Docker credential configuration if one
was used, and revoke the short-lived token:

```powershell
docker logout ghcr.io
Remove-Item Env:DOCKER_CONFIG
Remove-Item -LiteralPath $publishDockerConfig -Recurse -Force
```

The first push may create private GHCR packages. Change each package visibility
to public in GitHub only after verifying that its source, labels, tags, and
digest are the intended public release.

## 中文说明

公开镜像只包含项目自己的 API 和 Web 两个镜像；锁定 digest 的上游镜像只拉取，
不重新发布。准备阶段必须在独立、完整、仅有 GitHub HTTPS `origin` 的公开仓库
克隆中运行，并且此时不能登录 GHCR。脚本还会从固定 GitHub HTTPS 地址创建第二个
临时独立克隆，Docker 只使用该克隆作为构建上下文。脚本会先核对公开 `master`、
完整 CI 结果、跟踪文件秘密扫描和镜像身份，再生成不含注册表凭据的发布计划。

登录 GHCR 后不要再运行仓库中的 Python、npm、构建或 shell 代码，只执行事先
审核好的固定 Docker `tag`、`push`、远端 digest 检查和 `logout`；打 tag 时必须
以计划记录的本地 image ID 为源，不能依赖可变的本地 tag。标签只允许
`sha-<完整提交 SHA>` 与 `<版本>-<短 SHA>`，禁止 `latest` 和 `--all-tags`。
