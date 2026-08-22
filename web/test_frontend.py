#!/usr/bin/env python3
"""Frontend testing script using Playwright."""

import asyncio
from playwright.async_api import async_playwright
import os

FRONTEND_URL = "http://localhost:3000"
SCREENSHOT_DIR = "/workspace/train-factory/web/screenshots"

async def test_frontend():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            locale='zh-CN'
        )
        page = await context.new_page()

        # Collect errors
        errors = []
        page.on('console', lambda msg: errors.append(f"Console {msg.type}: {msg.text}") if msg.type == 'error' else None)
        page.on('pageerror', lambda err: errors.append(f"Page error: {err}"))

        print("=" * 60)
        print("TrainFactory Frontend Test")
        print("=" * 60)

        # Test 1: Training List Page
        print("\n[1/6] Testing Training List Page...")
        try:
            await page.goto(f"{FRONTEND_URL}/training", wait_until='networkidle', timeout=30000)
            await page.wait_for_timeout(2000)
            await page.screenshot(path=f"{SCREENSHOT_DIR}/01_training_list.png", full_page=True)

            # Check if page has content
            title = await page.locator('h4').first.text_content()
            print(f"  ✓ Page title: {title}")

            # Check if table exists
            table = page.locator('table')
            if await table.count() > 0:
                print("  ✓ Table found")
            else:
                print("  ✗ Table not found")
        except Exception as e:
            print(f"  ✗ Error: {e}")

        # Test 2: Training Create Page
        print("\n[2/6] Testing Training Create Page...")
        try:
            await page.goto(f"{FRONTEND_URL}/training/create", wait_until='networkidle', timeout=30000)
            await page.wait_for_timeout(2000)
            await page.screenshot(path=f"{SCREENSHOT_DIR}/02_training_create.png", full_page=True)

            # Check form elements
            form_items = await page.locator('.ant-form-item').count()
            print(f"  ✓ Form items found: {form_items}")

            # Check if model type select exists
            model_type = page.locator('text=模型类型')
            if await model_type.count() > 0:
                print("  ✓ Model type field found")
        except Exception as e:
            print(f"  ✗ Error: {e}")

        # Test 3: Datasets Page
        print("\n[3/6] Testing Datasets Page...")
        try:
            await page.goto(f"{FRONTEND_URL}/datasets", wait_until='networkidle', timeout=30000)
            await page.wait_for_timeout(2000)
            await page.screenshot(path=f"{SCREENSHOT_DIR}/03_datasets.png", full_page=True)

            title = await page.locator('h4').first.text_content()
            print(f"  ✓ Page title: {title}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

        # Test 4: Models Page
        print("\n[4/6] Testing Models Page...")
        try:
            await page.goto(f"{FRONTEND_URL}/models", wait_until='networkidle', timeout=30000)
            await page.wait_for_timeout(2000)
            await page.screenshot(path=f"{SCREENSHOT_DIR}/04_models.png", full_page=True)

            title = await page.locator('h4').first.text_content()
            print(f"  ✓ Page title: {title}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

        # Test 5: Deployments Page
        print("\n[5/6] Testing Deployments Page...")
        try:
            await page.goto(f"{FRONTEND_URL}/deployments", wait_until='networkidle', timeout=30000)
            await page.wait_for_timeout(2000)
            await page.screenshot(path=f"{SCREENSHOT_DIR}/05_deployments.png", full_page=True)

            title = await page.locator('h4').first.text_content()
            print(f"  ✓ Page title: {title}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

        # Test 6: Configs Page
        print("\n[6/6] Testing Configs Page...")
        try:
            await page.goto(f"{FRONTEND_URL}/configs", wait_until='networkidle', timeout=30000)
            await page.wait_for_timeout(2000)
            await page.screenshot(path=f"{SCREENSHOT_DIR}/06_configs.png", full_page=True)

            title = await page.locator('h4').first.text_content()
            print(f"  ✓ Page title: {title}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

        # Print errors
        if errors:
            print("\n" + "=" * 60)
            print("Console/Page Errors:")
            print("=" * 60)
            for err in errors[:20]:  # Limit to 20 errors
                print(f"  - {err}")

        await browser.close()

        print("\n" + "=" * 60)
        print(f"Screenshots saved to: {SCREENSHOT_DIR}")
        print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_frontend())
