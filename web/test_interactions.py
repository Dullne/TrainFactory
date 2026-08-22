#!/usr/bin/env python3
"""Frontend interaction testing script using Playwright."""

import asyncio
from playwright.async_api import async_playwright
import os

FRONTEND_URL = "http://localhost:3000"
SCREENSHOT_DIR = "/workspace/train-factory/web/screenshots/interactions"

async def test_interactions():
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
        print("TrainFactory Frontend Interaction Test")
        print("=" * 60)

        # ========================================
        # Test 1: Navigation
        # ========================================
        print("\n[1/8] Testing Navigation...")
        try:
            await page.goto(f"{FRONTEND_URL}/training", wait_until='networkidle', timeout=30000)
            await page.wait_for_timeout(1000)

            # Click each menu item
            menu_items = [
                ('数据集管理', '/datasets'),
                ('模型注册', '/models'),
                ('部署管理', '/deployments'),
                ('模型配置', '/configs'),
                ('训练管理', '/training'),
            ]

            for name, expected_path in menu_items:
                menu = page.locator(f'text={name}').first
                await menu.click()
                await page.wait_for_timeout(500)
                current_url = page.url
                if expected_path in current_url:
                    print(f"  ✓ Navigate to {name}")
                else:
                    print(f"  ✗ Failed to navigate to {name}, got {current_url}")

            await page.screenshot(path=f"{SCREENSHOT_DIR}/01_navigation.png")
        except Exception as e:
            print(f"  ✗ Error: {e}")

        # ========================================
        # Test 2: Create Training Task Form
        # ========================================
        print("\n[2/8] Testing Create Training Task...")
        try:
            await page.goto(f"{FRONTEND_URL}/training", wait_until='networkidle', timeout=30000)
            await page.wait_for_timeout(1000)

            # Click create button
            create_btn = page.locator('button:has-text("创建任务")')
            await create_btn.click()
            await page.wait_for_timeout(1000)
            await page.screenshot(path=f"{SCREENSHOT_DIR}/02_create_training_form.png")
            print("  ✓ Opened create training form")

            # Fill form fields
            # Task name
            task_name_input = page.locator('input[placeholder*="标识任务"]')
            await task_name_input.fill('test-embedding-task')
            print("  ✓ Filled task name")

            # Select model type (Embedding is default)
            model_type_select = page.locator('.ant-select').first
            await model_type_select.click()
            await page.wait_for_timeout(300)
            await page.locator('.ant-select-item-option:has-text("Embedding")').click()
            print("  ✓ Selected model type: Embedding")

            # Training method (SFT is default)
            await page.wait_for_timeout(300)

            # Model path
            model_path_input = page.locator('input[placeholder*="BAAI"]')
            await model_path_input.fill('BAAI/bge-base-zh-v1.5')
            print("  ✓ Filled model path")

            # Screenshot after filling
            await page.screenshot(path=f"{SCREENSHOT_DIR}/02_create_training_filled.png")
            print("  ✓ Form filled successfully")

            # Click cancel to go back
            cancel_btn = page.locator('button:has-text("取 消")')
            await cancel_btn.click()
            await page.wait_for_timeout(500)
            print("  ✓ Cancelled and returned to list")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            await page.screenshot(path=f"{SCREENSHOT_DIR}/02_error.png")

        # ========================================
        # Test 3: Dataset Management
        # ========================================
        print("\n[3/8] Testing Dataset Management...")
        try:
            await page.goto(f"{FRONTEND_URL}/datasets", wait_until='networkidle', timeout=30000)
            await page.wait_for_timeout(1000)

            # Click add dataset button
            add_btn = page.locator('button:has-text("添加数据集")')
            await add_btn.click()
            await page.wait_for_timeout(1000)

            # Check if modal opened
            modal = page.locator('.ant-modal')
            if await modal.count() > 0:
                print("  ✓ Opened add dataset modal")
                await page.screenshot(path=f"{SCREENSHOT_DIR}/03_add_dataset_modal.png")

                # Fill dataset form
                name_input = page.locator('.ant-modal input').first
                await name_input.fill('test-dataset')
                print("  ✓ Filled dataset name")

                # Select dataset type
                type_select = page.locator('.ant-modal .ant-select').first
                await type_select.click()
                await page.wait_for_timeout(300)
                # Select first option
                option = page.locator('.ant-select-item-option').first
                if await option.count() > 0:
                    await option.click()
                    print("  ✓ Selected dataset type")

                await page.screenshot(path=f"{SCREENSHOT_DIR}/03_dataset_form_filled.png")

                # Close modal
                close_btn = page.locator('.ant-modal button:has-text("取消")')
                if await close_btn.count() > 0:
                    await close_btn.click()
                else:
                    await page.keyboard.press('Escape')
                await page.wait_for_timeout(500)
                print("  ✓ Closed modal")
            else:
                print("  ✗ Modal not opened")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            await page.screenshot(path=f"{SCREENSHOT_DIR}/03_error.png")

        # ========================================
        # Test 4: Model Registry
        # ========================================
        print("\n[4/8] Testing Model Registry...")
        try:
            await page.goto(f"{FRONTEND_URL}/models", wait_until='networkidle', timeout=30000)
            await page.wait_for_timeout(1000)

            # Test filter dropdowns
            type_filter = page.locator('.ant-select:has-text("类型筛选")').first
            if await type_filter.count() > 0:
                await type_filter.click()
                await page.wait_for_timeout(300)
                await page.keyboard.press('Escape')
                print("  ✓ Type filter dropdown works")

            status_filter = page.locator('.ant-select:has-text("状态筛选")').first
            if await status_filter.count() > 0:
                await status_filter.click()
                await page.wait_for_timeout(300)
                await page.keyboard.press('Escape')
                print("  ✓ Status filter dropdown works")

            # Test refresh button
            refresh_btn = page.locator('button:has-text("刷新")')
            await refresh_btn.click()
            await page.wait_for_timeout(1000)
            print("  ✓ Refresh button works")

            await page.screenshot(path=f"{SCREENSHOT_DIR}/04_model_registry.png")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            await page.screenshot(path=f"{SCREENSHOT_DIR}/04_error.png")

        # ========================================
        # Test 5: Deployment Management
        # ========================================
        print("\n[5/8] Testing Deployment Management...")
        try:
            await page.goto(f"{FRONTEND_URL}/deployments", wait_until='networkidle', timeout=30000)
            await page.wait_for_timeout(1000)

            # Click create deployment button
            create_btn = page.locator('button:has-text("创建部署")')
            await create_btn.click()
            await page.wait_for_timeout(1000)

            # Check if modal opened
            modal = page.locator('.ant-modal')
            if await modal.count() > 0:
                print("  ✓ Opened create deployment modal")
                await page.screenshot(path=f"{SCREENSHOT_DIR}/05_create_deployment_modal.png")

                # Fill deployment name
                name_input = page.locator('.ant-modal input').first
                await name_input.fill('test-deployment')
                print("  ✓ Filled deployment name")

                await page.screenshot(path=f"{SCREENSHOT_DIR}/05_deployment_form_filled.png")

                # Close modal
                close_btn = page.locator('.ant-modal button:has-text("取消")')
                if await close_btn.count() > 0:
                    await close_btn.click()
                else:
                    await page.keyboard.press('Escape')
                await page.wait_for_timeout(500)
                print("  ✓ Closed modal")
            else:
                print("  ✗ Modal not opened")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            await page.screenshot(path=f"{SCREENSHOT_DIR}/05_error.png")

        # ========================================
        # Test 6: Model Config Management
        # ========================================
        print("\n[6/8] Testing Model Config Management...")
        try:
            await page.goto(f"{FRONTEND_URL}/configs", wait_until='networkidle', timeout=30000)
            await page.wait_for_timeout(1000)

            # Click add config button
            add_btn = page.locator('button:has-text("添加配置")')
            await add_btn.click()
            await page.wait_for_timeout(1000)

            # Check if modal opened
            modal = page.locator('.ant-modal')
            if await modal.count() > 0:
                print("  ✓ Opened add config modal")
                await page.screenshot(path=f"{SCREENSHOT_DIR}/06_add_config_modal.png")

                # Fill config name
                name_input = page.locator('.ant-modal input').first
                await name_input.fill('test-config')
                print("  ✓ Filled config name")

                # Select provider
                provider_select = page.locator('.ant-modal .ant-select').first
                await provider_select.click()
                await page.wait_for_timeout(300)
                option = page.locator('.ant-select-item-option').first
                if await option.count() > 0:
                    await option.click()
                    print("  ✓ Selected provider")

                await page.screenshot(path=f"{SCREENSHOT_DIR}/06_config_form_filled.png")

                # Close modal
                close_btn = page.locator('.ant-modal button:has-text("取消")')
                if await close_btn.count() > 0:
                    await close_btn.click()
                else:
                    await page.keyboard.press('Escape')
                await page.wait_for_timeout(500)
                print("  ✓ Closed modal")
            else:
                print("  ✗ Modal not opened")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            await page.screenshot(path=f"{SCREENSHOT_DIR}/06_error.png")

        # ========================================
        # Test 7: Training Task Form - Model Type Switching
        # ========================================
        print("\n[7/8] Testing Model Type Switching...")
        try:
            await page.goto(f"{FRONTEND_URL}/training/create", wait_until='networkidle', timeout=30000)
            await page.wait_for_timeout(1000)

            # Test different model types
            model_types = ['Embedding', 'Reranker', 'DecoderReranker']

            for model_type in model_types:
                # Click model type selector
                model_type_selects = page.locator('.ant-select')
                first_select = model_type_selects.first
                await first_select.click()
                await page.wait_for_timeout(300)

                # Select the model type
                option = page.locator(f'.ant-select-item-option:has-text("{model_type}")')
                if await option.count() > 0:
                    await option.click()
                    await page.wait_for_timeout(500)
                    print(f"  ✓ Switched to model type: {model_type}")
                    await page.screenshot(path=f"{SCREENSHOT_DIR}/07_model_type_{model_type.lower()}.png")
                else:
                    print(f"  - Model type {model_type} not found")
                    await page.keyboard.press('Escape')

        except Exception as e:
            print(f"  ✗ Error: {e}")
            await page.screenshot(path=f"{SCREENSHOT_DIR}/07_error.png")

        # ========================================
        # Test 8: Training Method Switching
        # ========================================
        print("\n[8/8] Testing Training Method Switching...")
        try:
            await page.goto(f"{FRONTEND_URL}/training/create", wait_until='networkidle', timeout=30000)
            await page.wait_for_timeout(1000)

            # Select Decoder Reranker model type to enable RL methods
            model_type_selects = page.locator('.ant-select')
            first_select = model_type_selects.first
            await first_select.click()
            await page.wait_for_timeout(300)
            decoder_option = page.locator('.ant-select-item-option:has-text("Decoder Reranker")')
            if await decoder_option.count() > 0:
                await decoder_option.click()
                await page.wait_for_timeout(500)
                print("  ✓ Selected Decoder Reranker model type")

            # Now test training methods
            training_methods = ['SFT', 'GRPO', 'DAPO']

            for method in training_methods:
                # Find the training method selector (second select)
                selects = page.locator('.ant-select')
                if await selects.count() >= 2:
                    method_select = selects.nth(1)
                    await method_select.click()
                    await page.wait_for_timeout(300)

                    option = page.locator(f'.ant-select-item-option:has-text("{method}")')
                    if await option.count() > 0:
                        await option.click()
                        await page.wait_for_timeout(500)
                        print(f"  ✓ Switched to training method: {method}")
                        await page.screenshot(path=f"{SCREENSHOT_DIR}/08_training_method_{method.lower()}.png")
                    else:
                        print(f"  - Training method {method} not found")
                        await page.keyboard.press('Escape')

        except Exception as e:
            print(f"  ✗ Error: {e}")
            await page.screenshot(path=f"{SCREENSHOT_DIR}/08_error.png")

        # Print errors
        if errors:
            print("\n" + "=" * 60)
            print("Console/Page Errors:")
            print("=" * 60)
            for err in errors[:20]:
                print(f"  - {err}")

        await browser.close()

        print("\n" + "=" * 60)
        print(f"Screenshots saved to: {SCREENSHOT_DIR}")
        print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_interactions())
