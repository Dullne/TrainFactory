import { test, expect } from '@playwright/test'
import { mockApi } from './helpers/mockApi'

test.beforeEach(async ({ page }) => {
  await mockApi(page)
})

test('training list renders', async ({ page }) => {
  await page.goto('/training')
  await expect(page.getByRole('heading', { name: '训练任务' })).toBeVisible()
  await expect(page.locator('table')).toBeVisible()
})

test('datasets page renders', async ({ page }) => {
  await page.goto('/datasets')
  await expect(page.getByRole('heading', { name: '数据集管理' })).toBeVisible()
  await expect(page.locator('table')).toBeVisible()
})

test('models page renders', async ({ page }) => {
  await page.goto('/models')
  await expect(page.getByRole('heading', { name: '模型列表' })).toBeVisible()
})

test('deployments page renders', async ({ page }) => {
  await page.goto('/deployments')
  await expect(page.getByRole('heading', { name: '部署列表' })).toBeVisible()
  await expect(page.locator('table')).toBeVisible()
})

test('configs page renders', async ({ page }) => {
  await page.goto('/configs')
  await expect(page.getByRole('heading', { name: '模型配置' })).toBeVisible()
})

test('resources page renders', async ({ page }) => {
  await page.goto('/resources')
  await expect(page.getByRole('heading', { name: '系统信息' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'GPU 监控' })).toBeVisible()
})

test('evaluations page renders', async ({ page }) => {
  await page.goto('/evaluations')
  await expect(page.getByRole('heading', { name: '评估任务' })).toBeVisible()
})
