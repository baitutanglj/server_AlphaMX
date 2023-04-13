const path = require('path');
module.exports = {
    entry: './app.js',
    output: {
        filename: 'bundle.js',
        path: path.join(__dirname, 'dist'),
    },
    target: 'node',
    mode: 'development',
    devServer: {
        static: "./dist"
    },
};
