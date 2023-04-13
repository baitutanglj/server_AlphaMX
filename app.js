const express = require('express')
const bodyParser = require('body-parser')
const cors = require('cors')
const app = express()
// app.engine('html', require('express-art-template'))
app.set('views', 'dist')
app.use('/', express.static('dist'))
app.use(bodyParser.urlencoded({ extended: false }));
//设置跨域
app.use(cors())

const generateRouter = require('./routes/structure_gen')
app.use('/generate', generateRouter)
const uploadRouter = require('./routes/upload')
app.use('/upload', uploadRouter)


const port = 8085
app.listen(port, ()=>{
    console.log('http://localhost:'+port, 'server start')
})

module.exports = app