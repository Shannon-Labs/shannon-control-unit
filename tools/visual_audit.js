const { test, expect, chromium } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

// Configure test settings
test.use({
  screenshot: 'only-on-failure',
  video: 'retain-on-failure',
});

const platforms = [
  { 
    name: 'GitHub', 
    url: 'https://github.com/Hmbown/shannon-control-unit',
    waitFor: '.markdown-body'
  },
  { 
    name: 'HuggingFace', 
    url: 'https://huggingface.co/hunterbown/shannon-control-unit',
    waitFor: '.prose'
  },
  { 
    name: 'Website', 
    url: 'https://shannonlabs.dev',
    waitFor: 'main'
  }
];

const viewports = [
  { name: 'Desktop', width: 1920, height: 1080 },
  { name: 'Tablet', width: 768, height: 1024 },
  { name: 'Mobile', width: 375, height: 667 }
];

// Issues tracker
const issues = {
  critical: [],
  high: [],
  medium: [],
  low: []
};

test.describe('Shannon Control Unit Visual Audit', () => {
  test.setTimeout(120000); // 2 minutes per test
  
  platforms.forEach(platform => {
    viewports.forEach(viewport => {
      test(`${platform.name} - ${viewport.name}`, async ({ browser }) => {
        const context = await browser.newContext({
          viewport: { width: viewport.width, height: viewport.height },
          userAgent: viewport.name === 'Mobile' 
            ? 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15'
            : undefined
        });
        
        const page = await context.newPage();
        
        // Enable console logging
        page.on('console', msg => {
          if (msg.type() === 'error') {
            issues.high.push({
              platform: platform.name,
              viewport: viewport.name,
              type: 'Console Error',
              message: msg.text()
            });
          }
        });
        
        // Track failed resources
        page.on('response', async response => {
          if (response.status() >= 400) {
            const url = response.url();
            if (url.includes('.png') || url.includes('.jpg') || url.includes('.jpeg') || url.includes('.svg')) {
              issues.critical.push({
                platform: platform.name,
                viewport: viewport.name,
                type: 'Broken Image',
                url: url,
                status: response.status()
              });
            }
          }
        });
        
        try {
          await page.goto(platform.url, { waitUntil: 'networkidle' });
          
          // Wait for main content
          await page.waitForSelector(platform.waitFor, { timeout: 30000 });
          
          // Take screenshot
          const screenshotPath = `assets/screenshots/${platform.name}_${viewport.name}.png`;
          await page.screenshot({ 
            path: screenshotPath,
            fullPage: true 
          });
          
          // Check for specific number formatting issues
          const content = await page.content();
          
          // Critical: Check for wrong number format (-0.244 instead of -6.2%)
          if (content.includes('-0.244') || content.includes('−0.244')) {
            const wrongNumbers = await page.locator('text=/-0\\.244|−0\\.244/').all();
            for (const element of wrongNumbers) {
              const bbox = await element.boundingBox();
              issues.critical.push({
                platform: platform.name,
                viewport: viewport.name,
                type: 'Wrong Number Format',
                description: 'Shows -0.244 instead of -6.2%',
                location: bbox
              });
            }
          }
          
          // Check for missing percentage signs
          const tables = await page.locator('table').all();
          for (const table of tables) {
            const text = await table.textContent();
            if (text.includes('BPT') || text.includes('improvement')) {
              // Check if percentages are properly formatted
              const hasPercentage = /−?\d+\.\d+%/.test(text);
              if (!hasPercentage && text.includes('improvement')) {
                issues.high.push({
                  platform: platform.name,
                  viewport: viewport.name,
                  type: 'Missing Percentage Sign',
                  description: 'Improvement values missing % sign'
                });
              }
            }
          }
          
          // Check for horizontal scrolling on mobile
          if (viewport.name === 'Mobile') {
            const bodyWidth = await page.evaluate(() => document.body.scrollWidth);
            const viewportWidth = viewport.width;
            if (bodyWidth > viewportWidth) {
              issues.high.push({
                platform: platform.name,
                viewport: viewport.name,
                type: 'Horizontal Scrolling',
                description: `Page width ${bodyWidth}px exceeds viewport ${viewportWidth}px`
              });
            }
          }
          
          // Check for overlapping elements
          const overlaps = await page.evaluate(() => {
            const elements = document.querySelectorAll('*');
            const overlapping = [];
            
            for (let i = 0; i < elements.length; i++) {
              const rect1 = elements[i].getBoundingClientRect();
              if (rect1.width === 0 || rect1.height === 0) continue;
              
              for (let j = i + 1; j < elements.length; j++) {
                const rect2 = elements[j].getBoundingClientRect();
                if (rect2.width === 0 || rect2.height === 0) continue;
                
                // Check if elements overlap
                if (!(rect1.right < rect2.left || 
                      rect2.right < rect1.left || 
                      rect1.bottom < rect2.top || 
                      rect2.bottom < rect1.top)) {
                  // Check if they're not parent-child
                  if (!elements[i].contains(elements[j]) && !elements[j].contains(elements[i])) {
                    overlapping.push({
                      element1: elements[i].tagName,
                      element2: elements[j].tagName
                    });
                  }
                }
              }
            }
            return overlapping;
          });
          
          if (overlaps.length > 0) {
            issues.medium.push({
              platform: platform.name,
              viewport: viewport.name,
              type: 'Overlapping Elements',
              count: overlaps.length
            });
          }
          
          // Check for broken links
          const links = await page.locator('a[href]').all();
          for (const link of links) {
            const href = await link.getAttribute('href');
            if (href && (href.startsWith('http://') || href.startsWith('https://'))) {
              try {
                const response = await page.request.head(href);
                if (response.status() >= 400) {
                  issues.high.push({
                    platform: platform.name,
                    viewport: viewport.name,
                    type: 'Broken Link',
                    url: href,
                    status: response.status()
                  });
                }
              } catch (e) {
                // Link check failed
              }
            }
          }
          
        } catch (error) {
          issues.critical.push({
            platform: platform.name,
            viewport: viewport.name,
            type: 'Page Load Error',
            error: error.message
          });
        } finally {
          await context.close();
        }
      });
    });
  });
  
  test.afterAll(async () => {
    // Generate report
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        critical: issues.critical.length,
        high: issues.high.length,
        medium: issues.medium.length,
        low: issues.low.length
      },
      issues: issues
    };
    
    fs.writeFileSync('visual_audit_report.json', JSON.stringify(report, null, 2));
    
    console.log('\n=== Visual Audit Summary ===');
    console.log(`Critical Issues: ${issues.critical.length}`);
    console.log(`High Priority: ${issues.high.length}`);
    console.log(`Medium Priority: ${issues.medium.length}`);
    console.log(`Low Priority: ${issues.low.length}`);
    
    if (issues.critical.length > 0) {
      console.log('\n⚠️ Critical Issues Found:');
      issues.critical.forEach(issue => {
        console.log(`- [${issue.platform}/${issue.viewport}] ${issue.type}: ${issue.description || issue.url || issue.error}`);
      });
    }
  });
});
