import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import {
  closeSync as fsCloseSync,
  fstatSync as fsFstatSync,
  openSync as fsOpenSync,
  readFileSync as fsReadFileSync,
  realpathSync as fsRealpathSync,
  statSync as fsStatSync,
} from 'node:fs'
import { gzipSync } from 'node:zlib'
import { afterEach, test } from 'node:test'
import { mkdir, mkdtemp, rm, symlink, writeFile } from 'node:fs/promises'
import { join, dirname } from 'node:path'
import { tmpdir } from 'node:os'
import { fileURLToPath } from 'node:url'

import * as bundleBudgetModule from './bundle-budget.mjs'
import {
  analyzeBundle,
  budgets,
  collectSynchronousClosure,
  evaluateBundleBudget,
  formatBundleBudgetReport,
} from './bundle-budget.mjs'

const DETAIL_KEY = 'src/pages/training/TrainingDetail.tsx'
const LOGIN_KEY = 'src/pages/auth/Login.tsx'
const SCRIPT_PATH = fileURLToPath(new URL('./bundle-budget.mjs', import.meta.url))
const fixtureRoots = []

afterEach(async () => {
  await Promise.all(
    fixtureRoots.splice(0).map((root) => rm(root, { recursive: true, force: true }))
  )
})

async function createFixture(manifest, assets = {}) {
  const root = await mkdtemp(join(tmpdir(), 'train-factory-bundle-budget-'))
  fixtureRoots.push(root)
  const distRoot = join(root, 'dist')
  const manifestPath = join(distRoot, '.vite', 'manifest.json')

  await mkdir(dirname(manifestPath), { recursive: true })
  await Promise.all(
    Object.entries(assets).map(async ([asset, contents]) => {
      const assetPath = join(distRoot, ...asset.split('/'))
      await mkdir(dirname(assetPath), { recursive: true })
      await writeFile(assetPath, contents)
    })
  )
  await writeFile(manifestPath, JSON.stringify(manifest))

  return { root, distRoot, manifestPath }
}

async function createRawManifest(contents) {
  const root = await mkdtemp(join(tmpdir(), 'train-factory-bundle-budget-'))
  fixtureRoots.push(root)
  const manifestPath = join(root, 'dist', '.vite', 'manifest.json')
  await mkdir(dirname(manifestPath), { recursive: true })
  await writeFile(manifestPath, contents)
  return { root, manifestPath }
}

function runCli(args = []) {
  return spawnSync(process.execPath, [SCRIPT_PATH, ...args], {
    encoding: 'utf8',
    windowsHide: true,
  })
}

function assertSanitizedCliFailure(result, expected, forbidden = []) {
  assert.equal(result.status, 1)
  assert.equal(result.signal, null)
  assert.equal(result.stdout, '')
  assert.equal(result.stderr, `${expected}\n`)
  assert.equal(result.stderr.includes('\r'), false)
  for (const value of forbidden) {
    assert.equal(result.stderr.includes(value), false)
  }
}

function isBundleError(code, message, forbidden = []) {
  return (error) => {
    assert.equal(error.code, code)
    assert.equal(error.message, message)
    for (const value of forbidden) {
      assert.equal(error.message.includes(value), false)
    }
    return true
  }
}

function validManifest(overrides = {}) {
  return {
    'src/main.tsx': {
      file: 'js/main.js',
      isEntry: true,
      imports: ['_shared.js'],
      dynamicImports: [LOGIN_KEY, DETAIL_KEY],
    },
    '_shared.js': {
      file: 'chunks/shared.js',
    },
    [LOGIN_KEY]: {
      file: 'chunks/login.js',
      isDynamicEntry: true,
      imports: ['_shared.js'],
    },
    [DETAIL_KEY]: {
      file: 'chunks/training-detail.js',
      isDynamicEntry: true,
      imports: ['_shared.js', '_detail-helper.js'],
    },
    '_detail-helper.js': {
      file: 'chunks/detail-helper.js',
    },
    ...overrides,
  }
}

const validAssets = {
  'js/main.js': 'entry-code',
  'chunks/shared.js': 'shared-code',
  'chunks/login.js': 'login-code',
  'chunks/training-detail.js': 'training-detail-code',
  'chunks/detail-helper.js': 'detail-helper-code',
}

test('CLI emits a stable usage error without arguments', () => {
  assertSanitizedCliFailure(runCli(), 'BUNDLE_BUDGET_USAGE: Bundle manifest path is required.')
})

test('CLI redacts an absolute missing manifest path and filesystem details', () => {
  const missingPath = join(tmpdir(), 'CANARY_SECRET-missing-bundle-manifest.json')

  assertSanitizedCliFailure(
    runCli([missingPath]),
    'BUNDLE_BUDGET_MANIFEST_READ_FAILED: Unable to read bundle manifest.',
    [missingPath, 'CANARY_SECRET', 'ENOENT']
  )
})

test('CLI rejects malformed JSON, null, and arrays with one generic error', async () => {
  const malformed = await createRawManifest('{"CANARY_SECRET\\r\\nFORGED:"')
  const nullManifest = await createRawManifest('null')
  const arrayManifest = await createRawManifest('[]')

  for (const manifestPath of [
    malformed.manifestPath,
    nullManifest.manifestPath,
    arrayManifest.manifestPath,
  ]) {
    assertSanitizedCliFailure(
      runCli([manifestPath]),
      'BUNDLE_BUDGET_MANIFEST_INVALID: Bundle manifest must be a JSON object.',
      [manifestPath, 'CANARY_SECRET', 'FORGED:']
    )
  }
})

test('CLI rejects control characters in every manifest key, import, and file without echoing them', async () => {
  const canary = 'CANARY_SECRET\r\nFORGED:'
  const manifests = [
    validManifest({
      'src/main.tsx': {
        file: `js/${canary}.js`,
        isEntry: true,
        imports: ['_shared.js'],
      },
    }),
    validManifest({
      [canary]: { file: 'chunks/unused.js' },
    }),
    validManifest({
      'src/main.tsx': {
        file: 'js/main.js',
        isEntry: true,
        imports: [canary],
      },
    }),
  ]

  for (const manifest of manifests) {
    const { manifestPath } = await createFixture(manifest, validAssets)
    assertSanitizedCliFailure(
      runCli([manifestPath]),
      'BUNDLE_BUDGET_MANIFEST_TEXT_INVALID: Bundle manifest contains unsafe text.',
      ['CANARY_SECRET', 'FORGED:', manifestPath]
    )
  }
})

test('CLI rejects Unicode line separators and bidi controls in every manifest key, import, and file', async () => {
  const unsafeCodePoints = [
    0x061c, 0x200e, 0x200f, 0x2028, 0x2029, 0x202a, 0x202b, 0x202c, 0x202d, 0x202e, 0x2066, 0x2067,
    0x2068, 0x2069,
  ]
  const manifestFactories = [
    (canary) => validManifest({ [canary]: { file: 'chunks/unused.js' } }),
    (canary) =>
      validManifest({
        '_unused-import.js': {
          file: 'chunks/unused.js',
          imports: [canary],
        },
      }),
    (canary) =>
      validManifest({
        '_unused-file.js': {
          file: `chunks/${canary}.js`,
        },
      }),
  ]

  for (const codePoint of unsafeCodePoints) {
    const unsafeCharacter = String.fromCodePoint(codePoint)
    const canary = `CANARY_SECRET${unsafeCharacter}FORGED:`
    for (const createManifest of manifestFactories) {
      const { manifestPath } = await createFixture(createManifest(canary), validAssets)
      assertSanitizedCliFailure(
        runCli([manifestPath]),
        'BUNDLE_BUDGET_MANIFEST_TEXT_INVALID: Bundle manifest contains unsafe text.',
        [manifestPath, canary, unsafeCharacter, 'CANARY_SECRET', 'FORGED:']
      )
    }
  }
})

test('CLI reports validated safe asset names and measurements for a real over-budget bundle', async () => {
  const { manifestPath } = await createFixture(validManifest(), {
    ...validAssets,
    'js/main.js': Buffer.alloc(800 * 1024, 'a'),
  })

  const result = runCli([manifestPath])

  assert.equal(result.status, 1)
  assert.equal(result.signal, null)
  assert.equal(result.stderr, '')
  assert.match(result.stdout, /Route-aware bundle budget report/)
  assert.match(result.stdout, /js\/main\.js: raw=819,200 B, gzip=[\d,]+ B/)
  assert.match(result.stdout, /initial chunk raw: actual=819,200 B, budget < 819,200 B/)
  assert.match(result.stdout, /FAIL: bundle budget exceeded\./)
  assert.equal(/[\u0000-\u001f\u007f-\u009f]/u.test(result.stdout.replaceAll('\n', '')), false)
})

test('fixed budgets are immutable and use the approved thresholds', () => {
  assert.deepEqual(budgets, {
    loginInitialGzipBytes: 350 * 1024,
    trainingDetailGzipBytes: 200 * 1024,
    initialChunkRawBytes: 800 * 1024,
  })
  assert.equal(Object.isFrozen(budgets), true)
})

test('collects the synchronous import closure once and ignores dynamic imports', () => {
  const manifest = validManifest({
    '_shared.js': {
      file: 'chunks/shared.js',
      imports: ['src/main.tsx', '_shared-alias.js'],
      dynamicImports: ['_never-initial.js'],
    },
    '_shared-alias.js': {
      file: 'chunks/shared.js',
    },
    '_never-initial.js': {
      file: 'chunks/dynamic-only.js',
    },
  })

  assert.deepEqual([...collectSynchronousClosure(manifest, ['src/main.tsx'])].sort(), [
    'chunks/shared.js',
    'js/main.js',
  ])
})

test('fails when a synchronous manifest import is missing', () => {
  const manifest = validManifest({
    '_shared.js': {
      file: 'chunks/shared.js',
      imports: ['_missing.js'],
    },
  })

  assert.throws(
    () => collectSynchronousClosure(manifest, ['src/main.tsx']),
    isBundleError('BUNDLE_BUDGET_MANIFEST_ENTRY_MISSING', 'Bundle manifest entry is missing.', [
      '_missing.js',
    ])
  )
})

test('fails closed when a manifest entry is inherited instead of owned', () => {
  const manifest = Object.create({
    'src/main.tsx': {
      file: 'js/main.js',
      isEntry: true,
    },
  })

  assert.throws(
    () => collectSynchronousClosure(manifest, ['src/main.tsx']),
    isBundleError('BUNDLE_BUDGET_MANIFEST_ENTRY_MISSING', 'Bundle manifest entry is missing.', [
      'src/main.tsx',
    ])
  )
})

test('fails closed unless imports is an array of strings', () => {
  for (const imports of ['_shared.js', {}, [42]]) {
    const manifest = validManifest({
      'src/main.tsx': {
        file: 'js/main.js',
        isEntry: true,
        imports,
      },
    })

    assert.throws(
      () => collectSynchronousClosure(manifest, ['src/main.tsx']),
      isBundleError('BUNDLE_BUDGET_MANIFEST_ENTRY_INVALID', 'Bundle manifest entry is invalid.', [
        'src/main.tsx',
      ])
    )
  }
})

test('fails closed with a redacted error unless every visited entry owns a .js file string', () => {
  const inheritedFileEntry = Object.create({ file: 'chunks/inherited.js' })
  const invalidEntries = [
    ['array entry', []],
    ['missing file', {}],
    ['non-string file', { file: 42 }],
    ['non-js file', { file: 'chunks/not-javascript.css' }],
    ['inherited file', inheritedFileEntry],
  ]

  for (const [label, entry] of invalidEntries) {
    const manifest = validManifest({ 'src/main.tsx': entry })

    assert.throws(
      () => collectSynchronousClosure(manifest, ['src/main.tsx']),
      isBundleError('BUNDLE_BUDGET_MANIFEST_ENTRY_INVALID', 'Bundle manifest entry is invalid.', [
        'src/main.tsx',
        'not-javascript.css',
        'inherited.js',
      ]),
      label
    )
  }
})

test('ignores inherited imports instead of traversing them', () => {
  const entry = Object.assign(Object.create({ imports: ['_inherited-missing.js'] }), {
    file: 'js/main.js',
    isEntry: true,
  })
  const manifest = validManifest({ 'src/main.tsx': entry })

  assert.deepEqual([...collectSynchronousClosure(manifest, ['src/main.tsx'])], ['js/main.js'])
})

test('measures entry and route-incremental assets using real raw and gzip bytes', async () => {
  const { manifestPath } = await createFixture(validManifest(), validAssets)

  const analysis = analyzeBundle(manifestPath)

  assert.equal(analysis.entryKey, 'src/main.tsx')
  assert.equal(analysis.loginKey, LOGIN_KEY)
  assert.equal(analysis.detailKey, DETAIL_KEY)
  assert.deepEqual(
    analysis.loginInitial.assets.map(({ asset }) => asset),
    ['chunks/login.js', 'chunks/shared.js', 'js/main.js']
  )
  assert.deepEqual(
    analysis.trainingDetail.assets.map(({ asset }) => asset),
    ['chunks/detail-helper.js', 'chunks/training-detail.js']
  )
  assert.equal(
    analysis.loginInitial.rawBytes,
    Buffer.byteLength(validAssets['js/main.js']) +
      Buffer.byteLength(validAssets['chunks/shared.js']) +
      Buffer.byteLength(validAssets['chunks/login.js'])
  )
  assert.equal(
    analysis.trainingDetail.gzipBytes,
    gzipSync(Buffer.from(validAssets['chunks/training-detail.js'])).byteLength +
      gzipSync(Buffer.from(validAssets['chunks/detail-helper.js'])).byteLength
  )
})

test('measures each asset once from one descriptor and one stable Buffer', async () => {
  const originalMain = Buffer.alloc(1024, 'a')
  const replacementMain = Buffer.allocUnsafe(1024)
  let state = 0x6d2b79f5
  for (let index = 0; index < replacementMain.length; index += 1) {
    state ^= state << 13
    state ^= state >>> 17
    state ^= state << 5
    replacementMain[index] = state & 0xff
  }
  assert.notEqual(gzipSync(originalMain).byteLength, gzipSync(replacementMain).byteLength)

  const manifest = validManifest({
    [LOGIN_KEY]: {
      file: 'chunks/login.js',
      isDynamicEntry: true,
      imports: ['_shared.js', '_login-detail-shared.js'],
    },
    [DETAIL_KEY]: {
      file: 'chunks/training-detail.js',
      isDynamicEntry: true,
      imports: ['_shared.js', '_detail-helper.js', '_login-detail-shared.js'],
    },
    '_login-detail-shared.js': {
      file: 'chunks/login-detail-shared.js',
    },
  })
  const { manifestPath } = await createFixture(manifest, {
    ...validAssets,
    'js/main.js': originalMain,
    'chunks/login-detail-shared.js': 'login-detail-shared-code',
  })
  const records = []
  const activeRecords = new Map()
  const realpathCalls = []
  const fsOps = {
    realpathSync(path) {
      realpathCalls.push(path)
      return fsRealpathSync(path)
    },
    openSync(path, flags) {
      const fd = fsOpenSync(path, flags)
      const record = { path, fd, fstat: 0, read: 0, stat: 0, close: 0 }
      records.push(record)
      activeRecords.set(fd, record)
      return fd
    },
    fstatSync(fd, options) {
      activeRecords.get(fd).fstat += 1
      return fsFstatSync(fd, options)
    },
    readFileSync(fd) {
      const record = activeRecords.get(fd)
      record.read += 1
      return record.path.endsWith(join('js', 'main.js')) ? replacementMain : fsReadFileSync(fd)
    },
    statSync(path, options) {
      const record = [...activeRecords.values()].find((candidate) => candidate.path === path)
      record.stat += 1
      return fsStatSync(path, options)
    },
    closeSync(fd) {
      const record = activeRecords.get(fd)
      record.close += 1
      activeRecords.delete(fd)
      return fsCloseSync(fd)
    },
  }

  const analysis = analyzeBundle(manifestPath, fsOps)
  const measuredMain = analysis.loginInitial.assets.find(({ asset }) => asset === 'js/main.js')

  assert.equal(measuredMain.rawBytes, replacementMain.byteLength)
  assert.equal(measuredMain.gzipBytes, gzipSync(replacementMain).byteLength)
  assert.equal(records.length, 6)
  assert.equal(new Set(records.map(({ path }) => path)).size, records.length)
  for (const record of records) {
    assert.equal(record.path, fsRealpathSync(record.path))
    assert.deepEqual(
      {
        fstat: record.fstat,
        read: record.read,
        stat: record.stat,
        close: record.close,
      },
      { fstat: 2, read: 1, stat: 1, close: 1 }
    )
  }
  assert.equal(activeRecords.size, 0)
  for (const record of records) {
    assert.equal(realpathCalls.filter((path) => fsRealpathSync(path) === record.path).length, 2)
  }
})

test('closes an opened asset descriptor when reading fails and redacts the cause', async () => {
  const { manifestPath } = await createFixture(validManifest(), validAssets)
  let opened = 0
  let closed = 0
  const fsOps = {
    realpathSync: fsRealpathSync,
    openSync(path, flags) {
      opened += 1
      return fsOpenSync(path, flags)
    },
    fstatSync: fsFstatSync,
    readFileSync() {
      throw new Error('CANARY_SECRET read failure')
    },
    statSync: fsStatSync,
    closeSync(fd) {
      closed += 1
      return fsCloseSync(fd)
    },
  }

  assert.throws(
    () => analyzeBundle(manifestPath, fsOps),
    isBundleError('BUNDLE_BUDGET_ASSET_ACCESS_FAILED', 'Unable to read a bundle asset.', [
      'CANARY_SECRET',
    ])
  )
  assert.equal(opened, 1)
  assert.equal(closed, 1)
})

test('fails closed and closes the descriptor when asset identity changes', async () => {
  const { manifestPath } = await createFixture(validManifest(), validAssets)
  let fstatCalls = 0
  let closed = 0
  const fsOps = {
    realpathSync: fsRealpathSync,
    openSync: fsOpenSync,
    fstatSync(fd, options) {
      const stats = fsFstatSync(fd, options)
      fstatCalls += 1
      if (fstatCalls !== 2) return stats
      return {
        isFile: () => stats.isFile(),
        dev: stats.dev,
        ino: stats.ino,
        size: stats.size + 1n,
        mtimeNs: stats.mtimeNs,
        ctimeNs: stats.ctimeNs,
      }
    },
    readFileSync: fsReadFileSync,
    statSync: fsStatSync,
    closeSync(fd) {
      closed += 1
      return fsCloseSync(fd)
    },
  }

  assert.throws(
    () => analyzeBundle(manifestPath, fsOps),
    isBundleError('BUNDLE_BUDGET_ASSET_UNSTABLE', 'Bundle asset changed while being measured.')
  )
  assert.equal(fstatCalls, 2)
  assert.equal(closed, 1)
})

test('rejects swap-out and swap-back when the opened fd differs from the restored pathname', async () => {
  const insideContents = Buffer.from('inside-asset')
  const outsideContents = Buffer.from('outside-data')
  assert.equal(insideContents.byteLength, outsideContents.byteLength)
  const uniformAssets = Object.fromEntries(
    Object.keys(validAssets).map((asset) => [asset, insideContents])
  )
  const { root, manifestPath } = await createFixture(validManifest(), uniformAssets)
  const outsidePath = join(root, 'CANARY_SECRET-outside.js')
  await writeFile(outsidePath, outsideContents)
  let opened = 0
  let closed = 0
  let pathnameStats = 0
  const fsOps = {
    realpathSync: fsRealpathSync,
    openSync(path, flags) {
      opened += 1
      return fsOpenSync(opened === 1 ? outsidePath : path, flags)
    },
    fstatSync: fsFstatSync,
    readFileSync: fsReadFileSync,
    statSync(path, options) {
      pathnameStats += 1
      return fsStatSync(path, options)
    },
    closeSync(fd) {
      closed += 1
      return fsCloseSync(fd)
    },
  }

  assert.throws(
    () => analyzeBundle(manifestPath, fsOps),
    isBundleError('BUNDLE_BUDGET_ASSET_UNSTABLE', 'Bundle asset changed while being measured.', [
      'CANARY_SECRET',
      outsidePath,
    ])
  )
  assert.equal(opened, 1)
  assert.equal(closed, 1)
  assert.equal(pathnameStats, 1)
})

test('fails analysis when the unique application entry has no own file', async () => {
  const manifest = validManifest()
  delete manifest['src/main.tsx'].file
  const { manifestPath } = await createFixture(manifest, validAssets)

  assert.throws(
    () => analyzeBundle(manifestPath),
    isBundleError('BUNDLE_BUDGET_MANIFEST_ENTRY_INVALID', 'Bundle manifest entry is invalid.', [
      'src/main.tsx',
    ])
  )
})

test('counts the entry and Login closures while subtracting only entry assets from TrainingDetail', async () => {
  const manifest = validManifest({
    [LOGIN_KEY]: {
      file: 'chunks/login.js',
      isDynamicEntry: true,
      imports: ['_shared.js', '_login-helper.js', '_login-detail-shared.js'],
    },
    [DETAIL_KEY]: {
      file: 'chunks/training-detail.js',
      isDynamicEntry: true,
      imports: ['_shared.js', '_detail-helper.js', '_login-detail-shared.js'],
    },
    '_login-helper.js': {
      file: 'chunks/login-helper.js',
    },
    '_login-detail-shared.js': {
      file: 'chunks/login-detail-shared.js',
    },
  })
  const { manifestPath } = await createFixture(manifest, {
    ...validAssets,
    'chunks/login-helper.js': 'login-helper-code',
    'chunks/login-detail-shared.js': 'login-detail-shared-code',
  })

  const analysis = analyzeBundle(manifestPath)

  assert.deepEqual(
    analysis.loginInitial.assets.map(({ asset }) => asset),
    [
      'chunks/login-detail-shared.js',
      'chunks/login-helper.js',
      'chunks/login.js',
      'chunks/shared.js',
      'js/main.js',
    ]
  )
  assert.deepEqual(
    analysis.trainingDetail.assets.map(({ asset }) => asset),
    ['chunks/detail-helper.js', 'chunks/login-detail-shared.js', 'chunks/training-detail.js']
  )
  assert.deepEqual(analysis.initialChunks, analysis.loginInitial.assets)
})

test('fails unless the manifest has exactly one application entry', async () => {
  const noEntry = validManifest({
    'src/main.tsx': { file: 'js/main.js' },
  })
  const twoEntries = validManifest({
    'src/other.tsx': { file: 'js/other.js', isEntry: true },
  })
  const noEntryFixture = await createFixture(noEntry, validAssets)
  const twoEntryFixture = await createFixture(twoEntries, {
    ...validAssets,
    'js/other.js': 'other',
  })

  assert.throws(
    () => analyzeBundle(noEntryFixture.manifestPath),
    isBundleError('BUNDLE_BUDGET_MANIFEST_ENTRY_INVALID', 'Bundle manifest entry is invalid.')
  )
  assert.throws(
    () => analyzeBundle(twoEntryFixture.manifestPath),
    isBundleError('BUNDLE_BUDGET_MANIFEST_ENTRY_INVALID', 'Bundle manifest entry is invalid.')
  )
})

test('fails when the exact TrainingDetail source key is absent', async () => {
  const manifest = validManifest()
  delete manifest[DETAIL_KEY]
  const { manifestPath } = await createFixture(manifest, validAssets)

  assert.throws(
    () => analyzeBundle(manifestPath),
    isBundleError('BUNDLE_BUDGET_MANIFEST_ENTRY_MISSING', 'Bundle manifest entry is missing.', [
      DETAIL_KEY,
    ])
  )
})

test('fails when the exact Login source key is absent', async () => {
  const manifest = validManifest()
  delete manifest[LOGIN_KEY]
  const { manifestPath } = await createFixture(manifest, validAssets)

  assert.throws(
    () => analyzeBundle(manifestPath),
    isBundleError('BUNDLE_BUDGET_MANIFEST_ENTRY_MISSING', 'Bundle manifest entry is missing.', [
      LOGIN_KEY,
    ])
  )
})

test('fails when Login is not emitted as a dynamic entry', async () => {
  const manifest = validManifest({
    [LOGIN_KEY]: {
      file: 'chunks/login.js',
      imports: ['_shared.js'],
    },
  })
  const { manifestPath } = await createFixture(manifest, validAssets)

  assert.throws(
    () => analyzeBundle(manifestPath),
    isBundleError('BUNDLE_BUDGET_MANIFEST_ENTRY_INVALID', 'Bundle manifest entry is invalid.')
  )
})

test('fails when TrainingDetail is not emitted as a dynamic entry', async () => {
  const manifest = validManifest({
    [DETAIL_KEY]: {
      file: 'chunks/training-detail.js',
      imports: ['_shared.js', '_detail-helper.js'],
    },
  })
  const { manifestPath } = await createFixture(manifest, validAssets)

  assert.throws(
    () => analyzeBundle(manifestPath),
    isBundleError('BUNDLE_BUDGET_MANIFEST_ENTRY_INVALID', 'Bundle manifest entry is invalid.')
  )
})

test('fails when a manifest asset does not exist', async () => {
  const assets = { ...validAssets }
  delete assets['chunks/shared.js']
  const { manifestPath } = await createFixture(validManifest(), assets)

  assert.throws(
    () => analyzeBundle(manifestPath),
    isBundleError('BUNDLE_BUDGET_ASSET_ACCESS_FAILED', 'Unable to read a bundle asset.', [
      'chunks/shared.js',
    ])
  )
})

test('rejects absolute, Windows, UNC, backslash, and escaping asset paths', async () => {
  const unsafeAssets = [
    '/absolute.js',
    'C:/outside.js',
    'C:outside.js',
    'D:chunks/outside.js',
    'z:relative.js',
    'C:\\outside.js',
    '//server/share/outside.js',
    '\\\\server\\share\\outside.js',
    'chunks\\backslash.js',
    '../outside.js',
  ]

  for (const unsafeAsset of unsafeAssets) {
    const manifest = validManifest({
      'src/main.tsx': { file: unsafeAsset, isEntry: true },
    })
    const { manifestPath } = await createFixture(manifest, validAssets)
    assert.throws(
      () => analyzeBundle(manifestPath),
      isBundleError(
        'BUNDLE_BUDGET_ASSET_PATH_INVALID',
        'Bundle manifest contains an invalid asset path.',
        [unsafeAsset]
      ),
      unsafeAsset
    )
  }
})

test('classifies Windows drive-relative asset paths as unsafe on every host platform', () => {
  assert.equal(typeof bundleBudgetModule.isSafeManifestAssetPath, 'function')
  for (const asset of ['C:outside.js', 'D:chunks/outside.js', 'z:relative.js']) {
    assert.equal(bundleBudgetModule.isSafeManifestAssetPath(asset), false, asset)
  }
})

test('allows a contained filename beginning with two dots', async () => {
  const manifest = validManifest({
    'src/main.tsx': { file: 'js/..safe.js', isEntry: true },
  })
  const { manifestPath } = await createFixture(manifest, {
    ...validAssets,
    'js/..safe.js': 'safe',
  })

  const analysis = analyzeBundle(manifestPath)

  assert.equal(
    analysis.loginInitial.assets.some(({ asset }) => asset === 'js/..safe.js'),
    true
  )
})

test('rejects a manifest asset whose real path escapes through a link', async () => {
  const { root, distRoot, manifestPath } = await createFixture(
    validManifest({
      'src/main.tsx': { file: 'linked/outside.js', isEntry: true },
    }),
    validAssets
  )
  const outsideRoot = join(root, 'outside')
  await mkdir(outsideRoot)
  await writeFile(join(outsideRoot, 'outside.js'), 'outside')
  await symlink(
    outsideRoot,
    join(distRoot, 'linked'),
    process.platform === 'win32' ? 'junction' : 'dir'
  )

  assert.throws(
    () => analyzeBundle(manifestPath),
    isBundleError(
      'BUNDLE_BUDGET_ASSET_PATH_INVALID',
      'Bundle manifest contains an invalid asset path.',
      ['linked/outside.js']
    )
  )
})

test('uses strict less-than comparisons so equality is over budget', () => {
  const analysis = {
    loginInitial: { gzipBytes: 10, assets: [] },
    trainingDetail: { gzipBytes: 20, assets: [] },
    initialChunks: [{ asset: 'js/main.js', rawBytes: 30, gzipBytes: 8 }],
  }

  const result = evaluateBundleBudget(analysis, {
    loginInitialGzipBytes: 10,
    trainingDetailGzipBytes: 20,
    initialChunkRawBytes: 30,
  })

  assert.equal(result.passed, false)
  assert.deepEqual(
    result.violations.map(({ name }) => name),
    ['login initial gzip', 'training detail incremental gzip', 'initial chunk raw']
  )
})

test('reports offending asset names, raw/gzip measurements, and budgets', () => {
  const analysis = {
    loginInitial: {
      rawBytes: 12,
      gzipBytes: 10,
      assets: [{ asset: 'js/main.js', rawBytes: 12, gzipBytes: 10 }],
    },
    trainingDetail: { rawBytes: 0, gzipBytes: 0, assets: [] },
    initialChunks: [{ asset: 'js/main.js', rawBytes: 12, gzipBytes: 10 }],
  }
  const limits = {
    loginInitialGzipBytes: 10,
    trainingDetailGzipBytes: 1,
    initialChunkRawBytes: 12,
  }
  const result = evaluateBundleBudget(analysis, limits)

  const report = formatBundleBudgetReport(analysis, result, limits)

  assert.match(report, /js\/main\.js/)
  assert.match(report, /raw=12 B/)
  assert.match(report, /gzip=10 B/)
  assert.match(report, /budget < 10 B/)
  assert.match(report, /budget < 12 B/)
})
