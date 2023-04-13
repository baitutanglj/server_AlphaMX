// const express = require('express');
// const router = express.Router();
// router.get('/', (req, res) => {
//     console.log(req.query);
//     res.send(req.query)
// });

// router.post('/', (req, res) => {
//     console.log(req);
//     res.json('ok')
// });
//
// module.exports = router;

const express = require('express')
const bodyParser = require('body-parser');
const router = express.Router()
const uptools = require('../public/javascripts/upfile')
// router.post('/', uptools.multer().single('file'), (req, res, next) =>{
//     const data = req.body
//     console.log(req.body);
//
//     res.json({'smiles':data.smiles, 'file_type':data.file_type, 'email': data.email})
// })
router.post('/', (req, res, next) =>{
    const data = req.body
    console.log(req.body);

    res.json({'smiles':data.smiles, 'file_type':data.file_type, 'email': data.email})
})

module.exports = router