const express = require('express')
const sd = require('silly-datetime');
const path = require('path');
const shortid = require('shortid');
const router = express.Router()
const workerProcess = require('../public/javascripts/process')
const mkdirsSync = require('../public/javascripts/mkdirs')

router.post('/',(req, res, next) => {
    const data = req.body
    var input
    var output
    if(data.file_type=='smiles'){
        input = data.smiles
        const day = sd.format(new Date(),'YYYYMMDD')
        const id = shortid.generate()
        const dir = path.join("/tmp", 'AlphaMX', day, id)
        mkdirsSync(dir)
        output = path.join(dir, 'output.sdf')
    }else {
        input = data.sdfPath
        output = path.join(path.dirname(input), 'output.sdf')

    }
    var pid = workerProcess(`bash structure_gen.sh "${input}" ${data.file_type} ${output}`,
        data.email, 'AlphaMX result', output)
    res.json({'output':output, 'file_type':data.file_type, 'input': input})
})

module.exports = router
