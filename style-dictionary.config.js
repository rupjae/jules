module.exports = {
  source: ['src/design-tokens/*.json'],
  platforms: {
    css: {
      transformGroup: 'css',
      buildPath: 'src/styles/',
      prefix: 'ctp',
      files: [{
        destination: 'catppuccin-macchiato.css',
        format: 'css/variables',
        options: {
          outputReferences: true,
          selector: ':root'
        }
      }]
    },
    ts: {
      transformGroup: 'js',
      buildPath: 'src/styles/',
      prefix: 'ctp',
      files: [{
        destination: 'catppuccin-macchiato.ts',
        format: 'javascript/es6'
      }]
    }
  }
};
