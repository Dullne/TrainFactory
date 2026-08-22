import type { FullConfig } from '@playwright/test'
import { runPlaywrightGlobalSetup } from '../../playwright.runtime.mjs'

export default async function globalSetup(config: FullConfig): Promise<void> {
  await runPlaywrightGlobalSetup(config.metadata)
}
