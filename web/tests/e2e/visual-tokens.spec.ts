import { test, expect } from '@playwright/test'
import { mockApi } from './helpers/mockApi'

for (const style of ['classic', 'workbench', 'studio']) {
  for (const mode of ['dark', 'light']) {
    test(`${style} ${mode}: primary actions and semantic statistics keep visual hierarchy and contrast`, async ({
      page,
    }) => {
      await page.addInitScript(
        ({ style, mode }) => {
          localStorage.setItem('tf_language', 'en')
          localStorage.setItem('tf_appearance:v1', JSON.stringify({ style, mode, accent: 'blue' }))
        },
        { style, mode }
      )
      await mockApi(page)
      await page.goto('/training')
      const create = page.locator('.training-list-toolbar-actions .ant-btn-primary')
      await expect(create).toBeVisible()
      const background = await create.evaluate(
        (element) => getComputedStyle(element).backgroundImage
      )
      if (style !== 'workbench') expect(background).toContain('linear-gradient')
      const contrasts = await page
        .locator('.stat-card .ant-statistic-content')
        .evaluateAll((elements) => {
          const canvas = document.createElement('canvas')
          canvas.width = canvas.height = 1
          const context = canvas.getContext('2d', { willReadFrequently: true })!
          const pixel = () => Array.from(context.getImageData(0, 0, 1, 1).data).slice(0, 3)
          const luminance = (rgb: number[]) =>
            rgb
              .map((value) => value / 255)
              .map((value) => (value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4))
              .reduce((sum, value, index) => sum + value * [0.2126, 0.7152, 0.0722][index], 0)
          return elements.map((element) => {
            const parents: Element[] = []
            for (let parent = element.parentElement; parent; parent = parent.parentElement)
              parents.push(parent)
            context.fillStyle = '#fff'
            context.fillRect(0, 0, 1, 1)
            for (const parent of parents.reverse()) {
              context.fillStyle = getComputedStyle(parent).backgroundColor
              context.fillRect(0, 0, 1, 1)
            }
            const background = luminance(pixel())
            context.fillStyle = getComputedStyle(element).color
            context.fillRect(0, 0, 1, 1)
            const foreground = luminance(pixel())
            return (
              (Math.max(foreground, background) + 0.05) / (Math.min(foreground, background) + 0.05)
            )
          })
        })
      expect(contrasts.length).toBe(4)
      for (const contrast of contrasts) expect(contrast).toBeGreaterThanOrEqual(4.5)
    })
  }
}
