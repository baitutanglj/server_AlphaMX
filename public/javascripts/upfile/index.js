var express = require('express')
var multer  = require('multer')
const router = express.Router()
const sd = require('silly-datetime');
const path = require('path');
const shortid = require('shortid');
const mkdirsSync = require('../mkdirs')

const uptools = {
    multer(){
        var storage = multer.diskStorage({
            //配置上传目录
            destination: (req, file, cb)=> {
                //1获取当前日期，eg：20201201
                //2按照日期生成文件存储目录
                const day = sd.format(new Date(),'YYYYMMDD')
                const id = shortid.generate()
                const dir = path.join("/tmp", 'AlphaMX', day, id)
                mkdirsSync(dir)
                cb(null, dir)//上传之前目录必须存在
            },
            //修改上传后的文件名
            filename: function (req, file, cb) {
                cb(null, file.originalname)
            }
        })
        var upload = multer({ storage: storage })
        return upload
    }
}
module.exports = uptools;