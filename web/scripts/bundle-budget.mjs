import { closeSync, fstatSync, openSync, readFileSync, realpathSync, statSync } from 'node:fs'
import { dirname, isAbsolute, relative, resolve, sep, win32 } from 'node:path'
import { fileURLToPath } from 'node:url'
import { gzipSync } from 'node:zlib'

const TRAINING_DETAIL_KEY = 'src/pages/training/TrainingDetail.tsx'
const LOGIN_KEY = 'src/pages/auth/Login.tsx'
const UNSAFE_TERMINAL_CHARACTERS =
  /[\u0000-\u001f\u007f-\u009f\u061c\u200e\u200f\u2028\u2029\u202a-\u202e\u2066-\u2069]/u

const errorDefinitions = Object.freeze({
  usage: ['BUNDLE_BUDGET_USAGE', 'Bundle manifest path is required.'],
  manifestRead: ['BUNDLE_BUDGET_MANIFEST_READ_FAILED', 'Unable to read bundle manifest.'],
  manifestInvalid: ['BUNDLE_BUDGET_MANIFEST_INVALID', 'Bundle manifest must be a JSON object.'],
  manifestText: ['BUNDLE_BUDGET_MANIFEST_TEXT_INVALID', 'Bundle manifest contains unsafe text.'],
  entryMissing: ['BUNDLE_BUDGET_MANIFEST_ENTRY_MISSING', 'Bundle manifest entry is missing.'],
  entryInvalid: ['BUNDLE_BUDGET_MANIFEST_ENTRY_INVALID', 'Bundle manifest entry is invalid.'],
  assetPath: [
    'BUNDLE_BUDGET_ASSET_PATH_INVALID',
    'Bundle manifest contains an invalid asset path.',
  ],
  assetAccess: ['BUNDLE_BUDGET_ASSET_ACCESS_FAILED', 'Unable to read a bundle asset.'],
  assetUnstable: ['BUNDLE_BUDGET_ASSET_UNSTABLE', 'Bundle asset changed while being measured.'],
  internal: ['BUNDLE_BUDGET_INTERNAL_ERROR', 'Bundle budget analysis failed.'],
})

const defaultMeasureFsOps = Object.freeze({
  closeSync,
  fstatSync,
  openSync,
  readFileSync,
  realpathSync,
  statSync,
})

export class BundleBudgetError extends Error {
  constructor(code, message) {
    super(message)
    this.name = 'BundleBudgetError'
    this.code = code
  }
}

function bundleBudgetError(definition) {
  const [code, message] = errorDefinitions[definition]
  return new BundleBudgetError(code, message)
}

function assertSafeManifestText(value) {
  if (UNSAFE_TERMINAL_CHARACTERS.test(value)) {
    throw bundleBudgetError('manifestText')
  }
}

function validateManifestText(manifest) {
  for (const [key, entry] of Object.entries(manifest)) {
    assertSafeManifestText(key)
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) continue

    if (Object.hasOwn(entry, 'file') && typeof entry.file === 'string') {
      assertSafeManifestText(entry.file)
    }
    if (Object.hasOwn(entry, 'imports') && Array.isArray(entry.imports)) {
      for (const imported of entry.imports) {
        if (typeof imported === 'string') assertSafeManifestText(imported)
      }
    }
  }
}

export const budgets = Object.freeze({
  loginInitialGzipBytes: 350 * 1024,
  trainingDetailGzipBytes: 200 * 1024,
  initialChunkRawBytes: 800 * 1024,
})

function isContained(root, candidate) {
  const candidateRelative = relative(root, candidate)
  return (
    candidateRelative !== '' &&
    candidateRelative !== '..' &&
    !candidateRelative.startsWith(`..${sep}`) &&
    !isAbsolute(candidateRelative)
  )
}

export function isSafeManifestAssetPath(asset) {
  return (
    typeof asset === 'string' &&
    asset.length > 0 &&
    !UNSAFE_TERMINAL_CHARACTERS.test(asset) &&
    !asset.includes('\\') &&
    !isAbsolute(asset) &&
    win32.parse(asset).root === ''
  )
}

function resolveManifestAsset(distRoot, realDistRoot, asset, fsOps) {
  if (!isSafeManifestAssetPath(asset)) {
    throw bundleBudgetError('assetPath')
  }

  const resolvedAsset = resolve(distRoot, asset)
  if (!isContained(distRoot, resolvedAsset)) {
    throw bundleBudgetError('assetPath')
  }

  let realAsset
  try {
    realAsset = fsOps.realpathSync(resolvedAsset)
  } catch {
    throw bundleBudgetError('assetAccess')
  }
  if (!isContained(realDistRoot, realAsset)) {
    throw bundleBudgetError('assetPath')
  }

  return { realAsset, resolvedAsset }
}

export function collectSynchronousClosure(manifest, startKeys) {
  const seenKeys = new Set()
  const assets = new Set()

  const visit = (key) => {
    if (seenKeys.has(key)) return

    if (!Object.hasOwn(manifest, key)) {
      throw bundleBudgetError('entryMissing')
    }
    const entry = manifest[key]
    if (!entry || typeof entry !== 'object') {
      throw bundleBudgetError('entryMissing')
    }
    if (
      Array.isArray(entry) ||
      !Object.hasOwn(entry, 'file') ||
      typeof entry.file !== 'string' ||
      entry.file.length === 0 ||
      !entry.file.endsWith('.js')
    ) {
      throw bundleBudgetError('entryInvalid')
    }
    const imports = Object.hasOwn(entry, 'imports') ? entry.imports : []
    if (!Array.isArray(imports) || !imports.every((key) => typeof key === 'string')) {
      throw bundleBudgetError('entryInvalid')
    }

    seenKeys.add(key)
    assets.add(entry.file)
    for (const imported of imports) {
      visit(imported)
    }
  }

  for (const key of startKeys) visit(key)
  return assets
}

function sameAssetIdentity(before, after) {
  return (
    before.dev === after.dev &&
    before.ino === after.ino &&
    before.size === after.size &&
    before.mtimeNs === after.mtimeNs &&
    before.ctimeNs === after.ctimeNs
  )
}

function measureAsset(distRoot, realDistRoot, asset, fsOps) {
  const { realAsset, resolvedAsset } = resolveManifestAsset(distRoot, realDistRoot, asset, fsOps)
  let fd
  try {
    fd = fsOps.openSync(realAsset, 'r')
  } catch {
    throw bundleBudgetError('assetAccess')
  }

  let measurement
  let failure
  try {
    const before = fsOps.fstatSync(fd, { bigint: true })
    if (!before.isFile()) throw bundleBudgetError('assetAccess')

    const contents = fsOps.readFileSync(fd)
    if (!Buffer.isBuffer(contents)) throw bundleBudgetError('assetAccess')

    const after = fsOps.fstatSync(fd, { bigint: true })
    let realAssetAfter
    try {
      realAssetAfter = fsOps.realpathSync(resolvedAsset)
    } catch {
      throw bundleBudgetError('assetUnstable')
    }
    if (!isContained(realDistRoot, realAssetAfter) || realAssetAfter !== realAsset) {
      throw bundleBudgetError('assetUnstable')
    }
    let pathnameStats
    try {
      pathnameStats = fsOps.statSync(realAssetAfter, { bigint: true })
    } catch {
      throw bundleBudgetError('assetUnstable')
    }
    if (
      !after.isFile() ||
      !pathnameStats.isFile() ||
      !sameAssetIdentity(before, after) ||
      !sameAssetIdentity(after, pathnameStats) ||
      before.size !== BigInt(contents.byteLength)
    ) {
      throw bundleBudgetError('assetUnstable')
    }

    measurement = {
      asset,
      rawBytes: contents.byteLength,
      gzipBytes: gzipSync(contents).byteLength,
    }
  } catch (error) {
    failure = error instanceof BundleBudgetError ? error : bundleBudgetError('assetAccess')
  } finally {
    try {
      fsOps.closeSync(fd)
    } catch {
      failure ??= bundleBudgetError('assetAccess')
    }
  }
  if (failure) throw failure
  return measurement
}

function measureAssets(distRoot, assets, fsOps) {
  let realDistRoot
  try {
    realDistRoot = fsOps.realpathSync(distRoot)
  } catch {
    throw bundleBudgetError('assetAccess')
  }
  return [...assets].sort().map((asset) => measureAsset(distRoot, realDistRoot, asset, fsOps))
}

function summarizeAssets(assets) {
  return {
    assets,
    rawBytes: assets.reduce((total, asset) => total + asset.rawBytes, 0),
    gzipBytes: assets.reduce((total, asset) => total + asset.gzipBytes, 0),
  }
}

export function analyzeBundle(manifestPath, measureFsOps = defaultMeasureFsOps) {
  const resolvedManifestPath = resolve(manifestPath)
  let manifestSource
  try {
    manifestSource = readFileSync(resolvedManifestPath, 'utf8')
  } catch {
    throw bundleBudgetError('manifestRead')
  }

  let manifest
  try {
    manifest = JSON.parse(manifestSource)
  } catch {
    throw bundleBudgetError('manifestInvalid')
  }
  if (!manifest || typeof manifest !== 'object' || Array.isArray(manifest)) {
    throw bundleBudgetError('manifestInvalid')
  }
  validateManifestText(manifest)
  const entryKeys = Object.entries(manifest)
    .filter(([, entry]) => entry?.isEntry === true)
    .map(([key]) => key)

  if (entryKeys.length !== 1) {
    throw bundleBudgetError('entryInvalid')
  }
  if (!Object.hasOwn(manifest, LOGIN_KEY)) {
    throw bundleBudgetError('entryMissing')
  }
  if (manifest[LOGIN_KEY]?.isDynamicEntry !== true) {
    throw bundleBudgetError('entryInvalid')
  }
  if (!Object.hasOwn(manifest, TRAINING_DETAIL_KEY)) {
    throw bundleBudgetError('entryMissing')
  }
  if (manifest[TRAINING_DETAIL_KEY]?.isDynamicEntry !== true) {
    throw bundleBudgetError('entryInvalid')
  }

  const entryKey = entryKeys[0]
  const entryClosure = collectSynchronousClosure(manifest, [entryKey])
  const loginClosure = collectSynchronousClosure(manifest, [LOGIN_KEY])
  const loginInitialAssets = new Set([...entryClosure, ...loginClosure])
  const detailClosure = collectSynchronousClosure(manifest, [TRAINING_DETAIL_KEY])
  const trainingDetailAssets = new Set(
    [...detailClosure].filter((asset) => !entryClosure.has(asset))
  )
  const distRoot = resolve(dirname(resolvedManifestPath), '..')
  const measuredAssets = measureAssets(
    distRoot,
    new Set([...loginInitialAssets, ...trainingDetailAssets]),
    measureFsOps
  )
  const initialChunks = measuredAssets.filter(({ asset }) => loginInitialAssets.has(asset))
  const measuredTrainingDetailAssets = measuredAssets.filter(({ asset }) =>
    trainingDetailAssets.has(asset)
  )

  return {
    entryKey,
    loginKey: LOGIN_KEY,
    detailKey: TRAINING_DETAIL_KEY,
    loginInitial: summarizeAssets(initialChunks),
    trainingDetail: summarizeAssets(measuredTrainingDetailAssets),
    initialChunks,
  }
}

export function evaluateBundleBudget(analysis, limits = budgets) {
  const violations = []

  if (!(analysis.loginInitial.gzipBytes < limits.loginInitialGzipBytes)) {
    violations.push({
      name: 'login initial gzip',
      actualBytes: analysis.loginInitial.gzipBytes,
      budgetBytes: limits.loginInitialGzipBytes,
      assets: analysis.loginInitial.assets,
    })
  }
  if (!(analysis.trainingDetail.gzipBytes < limits.trainingDetailGzipBytes)) {
    violations.push({
      name: 'training detail incremental gzip',
      actualBytes: analysis.trainingDetail.gzipBytes,
      budgetBytes: limits.trainingDetailGzipBytes,
      assets: analysis.trainingDetail.assets,
    })
  }
  for (const asset of analysis.initialChunks) {
    if (!(asset.rawBytes < limits.initialChunkRawBytes)) {
      violations.push({
        name: 'initial chunk raw',
        actualBytes: asset.rawBytes,
        budgetBytes: limits.initialChunkRawBytes,
        assets: [asset],
      })
    }
  }

  return {
    passed: violations.length === 0,
    violations,
  }
}

function formatBytes(bytes) {
  return `${bytes.toLocaleString('en-US')} B`
}

function formatAsset(asset) {
  return `  - ${asset.asset}: raw=${formatBytes(asset.rawBytes)}, gzip=${formatBytes(asset.gzipBytes)}`
}

export function formatBundleBudgetReport(analysis, result, limits = budgets) {
  const lines = [
    'Route-aware bundle budget report',
    `login initial: raw=${formatBytes(analysis.loginInitial.rawBytes)}, gzip=${formatBytes(analysis.loginInitial.gzipBytes)}, budget < ${formatBytes(limits.loginInitialGzipBytes)}`,
    ...analysis.loginInitial.assets.map(formatAsset),
    `training detail incremental: raw=${formatBytes(analysis.trainingDetail.rawBytes)}, gzip=${formatBytes(analysis.trainingDetail.gzipBytes)}, budget < ${formatBytes(limits.trainingDetailGzipBytes)}`,
    ...analysis.trainingDetail.assets.map(formatAsset),
    `initial chunk raw budget < ${formatBytes(limits.initialChunkRawBytes)}`,
    ...analysis.initialChunks.map(formatAsset),
    result.passed ? 'PASS: all bundle budgets are within limits.' : 'FAIL: bundle budget exceeded.',
  ]

  if (!result.passed) {
    lines.push('Violations:')
    for (const violation of result.violations) {
      lines.push(
        `  - ${violation.name}: actual=${formatBytes(violation.actualBytes)}, budget < ${formatBytes(violation.budgetBytes)}`
      )
      lines.push(...violation.assets.map(formatAsset))
    }
  }

  return lines.join('\n')
}

export function runBundleBudgetCli(manifestPath) {
  if (!manifestPath) {
    throw bundleBudgetError('usage')
  }
  const analysis = analyzeBundle(manifestPath)
  const result = evaluateBundleBudget(analysis)
  console.log(formatBundleBudgetReport(analysis, result))
  return result.passed ? 0 : 1
}

const invokedAsScript =
  process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)

if (invokedAsScript) {
  try {
    process.exitCode = runBundleBudgetCli(process.argv[2])
  } catch (error) {
    const publicError = error instanceof BundleBudgetError ? error : bundleBudgetError('internal')
    console.error(`${publicError.code}: ${publicError.message}`)
    process.exitCode = 1
  }
}
