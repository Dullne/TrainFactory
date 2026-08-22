const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  try {
    await page.goto('http://localhost:3000/deployments', { waitUntil: 'domcontentloaded', timeout: 15000 });
    await page.waitForTimeout(2000);
    await page.screenshot({ path: '../screenshots/frontend_check.png', fullPage: true });
    console.log('Frontend loaded');
  } catch (e) {
    console.log('Error:', e.message);
  }
  
  await browser.close();
})();
