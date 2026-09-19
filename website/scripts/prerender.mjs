import { createServer } from 'vite';
import { createElement } from 'react';
import { renderToString } from 'react-dom/server';
import { readFile, writeFile } from 'node:fs/promises';

// Preserve readable research content before JavaScript loads and for crawlers.
const server = await createServer({
  server: { middlewareMode: true, hmr: false, ws: false, watch: null },
  appType: 'custom',
});
try {
  const { default: Home } = await server.ssrLoadModule('/app/page.tsx');
  const markup = renderToString(createElement(Home));
  const template = await readFile('dist/index.html', 'utf8');
  if (!template.includes('<div id="root"></div>')) {
    throw new Error('Expected the empty React root in the Vite output.');
  }
  await writeFile('dist/index.html', template.replace('<div id="root"></div>', `<div id="root">${markup}</div>`));
  await writeFile('dist/.nojekyll', '');
  console.log('Prerendered the complete CausalWM research page.');
} finally {
  await server.close();
}
