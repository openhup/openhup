import adapter from '@sveltejs/adapter-static';

/**
 * Static adapter, output straight into the backend's `static/` directory.
 *
 * There is no SSR because there is no reason for it: this is a single-user dashboard behind
 * authentication on a LAN, and a static bundle means the API process serves the UI with no Node
 * runtime to install, update, or secure on someone's home server.
 *
 * @type {import('@sveltejs/kit').Config}
 */
export default {
  kit: {
    adapter: adapter({
      pages: '../backend/static',
      assets: '../backend/static',
      fallback: 'index.html', // SPA fallback: the router runs client-side
      precompress: true,
      strict: false
    }),
    // The API is same-origin in production; the dev server proxies to it (see vite.config.ts).
    paths: { base: '' },
    // A tight CSP is easy here because the UI needs no third-party anything.
    csp: {
      directives: {
        'default-src': ['self'],
        'img-src': ['self', 'data:'],
        // blob: lets server-side TTS audio play; speech is never fetched from a third party.
        'media-src': ['self', 'blob:'],
        'connect-src': ['self', 'ws:', 'wss:'],
        'frame-ancestors': ['none']
      }
    }
  }
};
