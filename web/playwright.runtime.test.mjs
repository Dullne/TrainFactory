import assert from 'node:assert/strict'
import test from 'node:test'

import {
  createWebServerConfig,
  extractPlaywrightRunId,
  resolvePlaywrightRuntime,
  resolvePlaywrightSetupMetadata,
  runPlaywrightGlobalSetup,
} from './playwright.runtime.mjs'

const metadataError = 'Playwright runtime metadata is invalid'
const managedSetupError = 'Unable to verify the managed Playwright server identity'

function assertRedactedError(error, expectedMessage, forbidden = []) {
  assert.ok(error instanceof Error)
  assert.equal(error.message, expectedMessage)
  const diagnostic = `${error.message}\n${error.stack ?? ''}`
  for (const value of forbidden) {
    assert.equal(diagnostic.includes(value), false)
  }
  return true
}

function managedMetadata(runId = 'managed-run-id') {
  return {
    trainFactoryE2E: {
      external: false,
      baseURL: 'http://127.0.0.1:41731',
      runId,
    },
  }
}

function setupResponse({
  body = '<html><head><meta name="train-factory-e2e-run" content="managed-run-id"></head></html>',
  ok = true,
  status = 200,
  redirected = false,
  url = 'http://127.0.0.1:41731/',
} = {}) {
  return {
    ok,
    status,
    redirected,
    url,
    async text() {
      return body
    },
  }
}

test('managed runtime defaults to an isolated server on port 41731', () => {
  const runtime = resolvePlaywrightRuntime({}, () => 'generated-run-id')

  assert.deepEqual(runtime, {
    external: false,
    baseURL: 'http://127.0.0.1:41731',
    port: 41731,
    runId: 'generated-run-id',
  })
  assert.equal(runtime.runId.length > 0, true)
})

test('managed runtime generates a nonempty run id by default', () => {
  const runtime = resolvePlaywrightRuntime({})

  assert.equal(runtime.external, false)
  assert.equal(typeof runtime.runId, 'string')
  assert.equal(runtime.runId.length > 0, true)
})

test('managed runtime accepts the inclusive port boundaries', () => {
  assert.equal(
    resolvePlaywrightRuntime({ PLAYWRIGHT_PORT: '1' }, () => 'run-one').port,
    1
  )
  assert.equal(
    resolvePlaywrightRuntime({ PLAYWRIGHT_PORT: '65535' }, () => 'run-max').port,
    65535
  )
})

for (const port of ['0', '65536', '1.5', 'not-a-number', '']) {
  test(`managed runtime rejects invalid port ${JSON.stringify(port)}`, () => {
    assert.throws(
      () => resolvePlaywrightRuntime({ PLAYWRIGHT_PORT: port }),
      /PLAYWRIGHT_PORT must be an integer between 1 and 65535/
    )
  })
}

test('managed web server is strict, never reused, and receives only its scoped env', () => {
  const runtime = resolvePlaywrightRuntime(
    { PLAYWRIGHT_PORT: '43220' },
    () => 'managed-run-id'
  )

  assert.deepEqual(createWebServerConfig(runtime), {
    command: 'npm run dev -- --host 127.0.0.1 --port 43220 --strictPort',
    url: 'http://127.0.0.1:43220',
    env: {
      VITE_API_BASE_URL: '/api',
      VITE_AUTH_LOGOUT_TIMEOUT: '1000',
      VITE_E2E_RUN_ID: 'managed-run-id',
    },
    reuseExistingServer: false,
    timeout: 120000,
  })
})

test('runtime uses the injected run-id factory exactly once', () => {
  let calls = 0
  const runtime = resolvePlaywrightRuntime({}, () => {
    calls += 1
    return 'injected-run-id'
  })

  assert.equal(calls, 1)
  assert.equal(runtime.runId, 'injected-run-id')
})

test('external mode requires its flag and base URL together', () => {
  assert.throws(
    () => resolvePlaywrightRuntime({ PLAYWRIGHT_USE_EXTERNAL_SERVER: '1' }),
    /External mode requires PLAYWRIGHT_USE_EXTERNAL_SERVER=1 and PLAYWRIGHT_BASE_URL together/
  )
  assert.throws(
    () => resolvePlaywrightRuntime({ PLAYWRIGHT_BASE_URL: 'https://example.invalid' }),
    /External mode requires PLAYWRIGHT_USE_EXTERNAL_SERVER=1 and PLAYWRIGHT_BASE_URL together/
  )
})

test('external runtime keeps only a credential-free HTTP origin and omits webServer', () => {
  const runtime = resolvePlaywrightRuntime({
    PLAYWRIGHT_USE_EXTERNAL_SERVER: '1',
    PLAYWRIGHT_BASE_URL: 'https://example.invalid:8443/some/path?token=discard-me#fragment',
  })

  assert.deepEqual(runtime, {
    external: true,
    baseURL: 'https://example.invalid:8443',
    runId: null,
  })
  assert.equal(createWebServerConfig(runtime), undefined)
})

for (const baseURL of [
  'ftp://example.invalid/path',
  'file:///tmp/train-factory',
  'https://operator@example.invalid/path',
  'https://operator:password@example.invalid/path',
]) {
  test(`external runtime rejects unsupported or credentialed URL ${JSON.stringify(baseURL)}`, () => {
    assert.throws(
      () =>
        resolvePlaywrightRuntime({
          PLAYWRIGHT_USE_EXTERNAL_SERVER: '1',
          PLAYWRIGHT_BASE_URL: baseURL,
        }),
      /PLAYWRIGHT_BASE_URL must be an HTTP\(S\) URL without credentials/
    )
  })
}

test('invalid external URL errors redact credentials, query, original URL, and canary secret', () => {
  const canary = 'CANARY-runtime-secret-6e01c7'
  const originalURL = `https://operator:${canary}@exa mple.invalid/path?token=${canary}`

  let failure
  try {
    resolvePlaywrightRuntime({
      PLAYWRIGHT_USE_EXTERNAL_SERVER: '1',
      PLAYWRIGHT_BASE_URL: originalURL,
    })
  } catch (error) {
    failure = error
  }

  assert.ok(failure instanceof Error)
  const diagnostic = `${failure.message}\n${failure.stack ?? ''}`
  assert.match(failure.message, /PLAYWRIGHT_BASE_URL must be a valid HTTP\(S\) URL without credentials/)
  assert.equal(diagnostic.includes(canary), false)
  assert.equal(diagnostic.includes(originalURL), false)
  assert.equal(diagnostic.includes('operator:'), false)
  assert.equal(diagnostic.includes('?token='), false)
})

for (const flag of ['', '0', 'true', ' 1 ']) {
  test(`external flag rejects explicit invalid value ${JSON.stringify(flag)}`, () => {
    assert.throws(
      () => resolvePlaywrightRuntime({ PLAYWRIGHT_USE_EXTERNAL_SERVER: flag }),
      /PLAYWRIGHT_USE_EXTERNAL_SERVER must be unset or exactly 1/
    )
  })
}

test('explicit empty external URL is invalid when external mode is enabled', () => {
  assert.throws(
    () =>
      resolvePlaywrightRuntime({
        PLAYWRIGHT_USE_EXTERNAL_SERVER: '1',
        PLAYWRIGHT_BASE_URL: '',
      }),
    /PLAYWRIGHT_BASE_URL must be a valid HTTP\(S\) URL without credentials/
  )
})

test('an explicitly present base URL still requires the external flag', () => {
  assert.throws(
    () => resolvePlaywrightRuntime({ PLAYWRIGHT_BASE_URL: '' }),
    /External mode requires PLAYWRIGHT_USE_EXTERNAL_SERVER=1 and PLAYWRIGHT_BASE_URL together/
  )
})

for (const [label, generated, forbidden] of [
  ['empty', '', []],
  ['blank', '   ', []],
  [
    'non-string',
    { value: 'CANARY-non-string-run-id-e258' },
    ['CANARY-non-string-run-id-e258'],
  ],
]) {
  test(`managed runtime rejects a ${label} generated run id after one factory call`, () => {
    let calls = 0
    let failure
    try {
      resolvePlaywrightRuntime({}, () => {
        calls += 1
        return generated
      })
    } catch (error) {
      failure = error
    }

    assert.equal(calls, 1)
    assertRedactedError(
      failure,
      'Playwright run ID generator must return a nonempty string',
      forbidden
    )
  })
}

test('managed runtime redacts errors thrown by the run-id factory', () => {
  const canary = 'CANARY-run-id-factory-error-2956'
  let calls = 0
  let failure
  try {
    resolvePlaywrightRuntime({}, () => {
      calls += 1
      throw new Error(canary)
    })
  } catch (error) {
    failure = error
  }

  assert.equal(calls, 1)
  assertRedactedError(
    failure,
    'Playwright run ID generator must return a nonempty string',
    [canary]
  )
})

test('setup metadata accepts only normalized external and managed origins', () => {
  assert.deepEqual(
    resolvePlaywrightSetupMetadata({
      trainFactoryE2E: {
        external: true,
        baseURL: 'https://example.invalid:8443',
        runId: null,
      },
    }),
    {
      external: true,
      baseURL: 'https://example.invalid:8443',
      runId: null,
    }
  )
  assert.deepEqual(resolvePlaywrightSetupMetadata(managedMetadata()), {
    external: false,
    baseURL: 'http://127.0.0.1:41731',
    runId: 'managed-run-id',
  })
})

for (const [label, metadata] of [
  ['missing metadata', undefined],
  ['array metadata', []],
  ['missing runtime', {}],
  ['array runtime', { trainFactoryE2E: [] }],
  [
    'non-boolean external marker',
    {
      trainFactoryE2E: {
        external: 'false',
        baseURL: 'https://example.invalid',
        runId: null,
      },
    },
  ],
  [
    'external run id',
    {
      trainFactoryE2E: {
        external: true,
        baseURL: 'https://example.invalid',
        runId: 'unexpected',
      },
    },
  ],
  [
    'blank managed run id',
    {
      trainFactoryE2E: {
        external: false,
        baseURL: 'http://127.0.0.1:41731',
        runId: '   ',
      },
    },
  ],
  [
    'credentialed origin',
    {
      trainFactoryE2E: {
        external: true,
        baseURL: 'https://operator:password@example.invalid',
        runId: null,
      },
    },
  ],
  [
    'origin with path and query',
    {
      trainFactoryE2E: {
        external: true,
        baseURL: 'https://example.invalid/path?token=secret',
        runId: null,
      },
    },
  ],
]) {
  test(`setup metadata rejects ${label} with a generic error`, () => {
    assert.throws(
      () => resolvePlaywrightSetupMetadata(metadata),
      (error) => assertRedactedError(error, metadataError)
    )
  })
}

test('setup metadata failure redacts the original URL, credentials, query, and canary', () => {
  const canary = 'CANARY-setup-metadata-secret-e774'
  const originalURL = `https://operator:${canary}@example.invalid/path?token=${canary}`

  assert.throws(
    () =>
      resolvePlaywrightSetupMetadata({
        trainFactoryE2E: {
          external: 'false',
          baseURL: originalURL,
          runId: null,
        },
      }),
    (error) =>
      assertRedactedError(error, metadataError, [
        canary,
        originalURL,
        'operator:',
        '?token=',
      ])
  )
})

test('marker parser supports exact attributes in either order with whitespace', () => {
  assert.equal(
    extractPlaywrightRunId(
      '<html><head><meta name="train-factory-e2e-run" content="run-one"></head></html>'
    ),
    'run-one'
  )
  assert.equal(
    extractPlaywrightRunId(
      `<html><head><meta   content = 'run-two'   name = "train-factory-e2e-run" ></head></html>`
    ),
    'run-two'
  )
  assert.equal(
    extractPlaywrightRunId(
      '<html><head><base href="/"><link rel="icon" href="/mark.svg"><meta name="train-factory-e2e-run" content="run-three"></head><body></body></html>'
    ),
    'run-three'
  )
})

for (const [label, html] of [
  [
    'data attribute lookalikes',
    '<meta data-name="train-factory-e2e-run" data-content="forged-run">',
  ],
  [
    'duplicate marker elements',
    '<meta name="train-factory-e2e-run" content="one"><meta name="train-factory-e2e-run" content="two">',
  ],
  [
    'duplicate name attributes',
    '<meta name="train-factory-e2e-run" name="train-factory-e2e-run" content="one">',
  ],
  [
    'duplicate content attributes',
    '<meta name="train-factory-e2e-run" content="one" content="two">',
  ],
  ['missing content', '<meta name="train-factory-e2e-run">'],
  ['empty content', '<meta name="train-factory-e2e-run" content="">'],
  ['blank content', '<meta name="train-factory-e2e-run" content="   ">'],
]) {
  test(`marker parser rejects ${label}`, () => {
    const document = `<html><head>${html}</head><body></body></html>`
    assert.throws(() => extractPlaywrightRunId(document), /Playwright run marker is invalid/)
  })
}

for (const [label, html] of [
  [
    'meta-data tag lookalike',
    '<html><head><meta-data name="train-factory-e2e-run" content="forged"></meta-data></head></html>',
  ],
  [
    'namespaced meta tag lookalike',
    '<html><head><meta:shadow name="train-factory-e2e-run" content="forged"></meta:shadow></head></html>',
  ],
  [
    'slash-delimited meta tag lookalike',
    '<html><head><meta/name="train-factory-e2e-run" content="forged"></head></html>',
  ],
  [
    'HTML comment marker',
    '<html><head><!-- <meta name="train-factory-e2e-run" content="forged"> --></head></html>',
  ],
  [
    'script string marker',
    `<html><head><script>const marker = '<meta name="train-factory-e2e-run" content="forged">'</script></head></html>`,
  ],
  [
    'style text marker',
    `<html><head><style>.x::after { content: '<meta name="train-factory-e2e-run" content="forged">'; }</style></head></html>`,
  ],
  [
    'textarea text marker',
    '<html><head><textarea><meta name="train-factory-e2e-run" content="forged"></textarea></head></html>',
  ],
  [
    'template marker',
    '<html><head><template><meta name="train-factory-e2e-run" content="forged"></template></head></html>',
  ],
  [
    'noscript marker',
    '<html><head><noscript><meta name="train-factory-e2e-run" content="forged"></noscript></head></html>',
  ],
  [
    'body marker',
    '<html><head></head><body><meta name="train-factory-e2e-run" content="forged"></body></html>',
  ],
  [
    'head after body content already started',
    '<html><div></div><head><meta name="train-factory-e2e-run" content="forged"></head></html>',
  ],
  [
    'head after non-whitespace text already started',
    '<html>body text<head><meta name="train-factory-e2e-run" content="forged"></head></html>',
  ],
  [
    'marker after non-whitespace text inside head',
    '<html><head>body text<meta name="train-factory-e2e-run" content="forged"></head></html>',
  ],
  [
    'marker after body content starts inside head',
    '<html><head><div><meta name="train-factory-e2e-run" content="forged"></div></head></html>',
  ],
  [
    'marker inside foreign SVG content',
    '<html><head><svg><meta name="train-factory-e2e-run" content="forged"></svg></head></html>',
  ],
  [
    'marker after an unexpected end tag inside head',
    '<html><head></div><meta name="train-factory-e2e-run" content="forged"></head></html>',
  ],
  [
    'marker after an unclosed comment',
    '<html><head><!-- never closed <meta name="train-factory-e2e-run" content="forged"></head></html>',
  ],
  [
    'marker after an unclosed raw-text element',
    '<html><head><script>never closed <meta name="train-factory-e2e-run" content="forged"></head></html>',
  ],
]) {
  test(`marker parser rejects ${label}`, () => {
    assert.throws(() => extractPlaywrightRunId(html), /Playwright run marker is invalid/)
  })
}

for (const [label, invalidStartToken] of [
  ['at-sign tag suffix', '<div@>'],
  ['dot tag suffix', '<x.foo>'],
  ['dollar tag suffix', '<x$foo>'],
  ['bracket tag suffix', '<x[foo]>'],
  ['numeric tag name', '<123>'],
  ['dollar tag name', '<$>'],
  ['underscore tag name', '<_x>'],
  ['double equals attribute', '<div foo==bar>'],
  ['nonterminal self-closing slash', '<div / junk>'],
  ['backtick attribute prefix', '<div `x>'],
  ['less-than attribute prefix', '<div <x>'],
  ['dot meta suffix', '<meta.foo>'],
]) {
  for (const position of ['before head', 'inside head']) {
    test(`marker parser fails closed for ${label} ${position}`, () => {
      const marker = '<meta name="train-factory-e2e-run" content="forged">'
      const html =
        position === 'before head'
          ? `<html>${invalidStartToken}<head>${marker}</head></html>`
          : `<html><head>${invalidStartToken}${marker}</head></html>`

      assert.throws(() => extractPlaywrightRunId(html), /Playwright run marker is invalid/)
    })
  }
}

test('external setup writes only the validated origin and does not fetch', async () => {
  let fetchCalls = 0
  let output = ''
  await runPlaywrightGlobalSetup(
    {
      trainFactoryE2E: {
        external: true,
        baseURL: 'https://example.invalid:8443',
        runId: null,
      },
    },
    {
      async fetch() {
        fetchCalls += 1
        throw new Error('external setup must not fetch')
      },
      write(value) {
        output += value
      },
    }
  )

  assert.equal(fetchCalls, 0)
  assert.equal(
    output,
    '[TrainFactory E2E] external=true origin=https://example.invalid:8443; current source was not verified\n'
  )
})

test('setup rejects forged false metadata without writing credential or query canaries', async () => {
  const canary = 'CANARY-forged-external-false-33bd'
  const originalURL = `https://operator:${canary}@example.invalid/path?token=${canary}`
  let output = ''

  await assert.rejects(
    () =>
      runPlaywrightGlobalSetup(
        {
          trainFactoryE2E: {
            external: 'false',
            baseURL: originalURL,
            runId: null,
          },
        },
        { write: (value) => (output += value) }
      ),
    (error) =>
      assertRedactedError(error, metadataError, [
        canary,
        originalURL,
        'operator:',
        '?token=',
      ])
  )
  assert.equal(output, '')
})

test('managed setup requests redirect errors and verifies the exact run marker', async () => {
  let fetchArguments
  let output = ''
  await runPlaywrightGlobalSetup(managedMetadata(), {
    async fetch(...args) {
      fetchArguments = args
      return setupResponse()
    },
    write(value) {
      output += value
    },
  })

  assert.deepEqual(fetchArguments, [
    'http://127.0.0.1:41731',
    { redirect: 'error' },
  ])
  assert.equal(
    output,
    '[TrainFactory E2E] external=false origin=http://127.0.0.1:41731; current source run verified\n'
  )
})

for (const [label, response] of [
  ['302 response', setupResponse({ ok: false, status: 302 })],
  ['redirected response', setupResponse({ redirected: true })],
  ['cross-origin response', setupResponse({ url: 'https://other.invalid/' })],
  ['invalid response URL', setupResponse({ url: 'not a valid URL' })],
]) {
  test(`managed setup rejects ${label} with a generic error`, async () => {
    let output = ''
    await assert.rejects(
      () =>
        runPlaywrightGlobalSetup(managedMetadata(), {
          fetch: async () => response,
          write: (value) => (output += value),
        }),
      (error) => assertRedactedError(error, managedSetupError)
    )
    assert.equal(output, '')
  })
}

test('managed setup redacts fetch errors and writes nothing', async () => {
  const canary = 'CANARY-fetch-error-26de'
  let output = ''

  await assert.rejects(
    () =>
      runPlaywrightGlobalSetup(managedMetadata(), {
        fetch: async () => {
          throw new Error(`network failure ${canary}`)
        },
        write: (value) => (output += value),
      }),
    (error) => assertRedactedError(error, managedSetupError, [canary, 'network failure'])
  )
  assert.equal(output, '')
})

test('managed setup redacts an unexpected response body and writes nothing', async () => {
  const canary = 'CANARY-response-body-47c9'
  const responseBody = `<html><body>${canary}?token=${canary}</body></html>`
  let output = ''

  await assert.rejects(
    () =>
      runPlaywrightGlobalSetup(managedMetadata(), {
        fetch: async () => setupResponse({ body: responseBody }),
        write: (value) => (output += value),
      }),
    (error) =>
      assertRedactedError(error, managedSetupError, [canary, responseBody, '?token='])
  )
  assert.equal(output, '')
})
