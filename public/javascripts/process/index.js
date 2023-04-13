const child_process = require('child_process')
const emailcom = require('../email')
var wokrPath = process.cwd();
wokrPath = wokrPath + '/public/AlphaMX'
function workerProcess(mycmd,toemail,subject,resultPath) {
    var exec = child_process.exec(
        mycmd,
        {cwd:wokrPath,maxBuffer:1024*1024*1024},
        function( error, stdout, stderr) {
            // console.log('pid:' + exec.pid);
            if (error) {
                console.log(error.stack)
                console.log('Error code: ' + error.code)
                console.log('Signal received: ' + error.signal)
            }
            // console.log('stdout: '+stdout);
            emailcom(toemail, subject, resultPath)
        }
    )
    exec.on('exit', function(code) {
    })
    return exec.pid
}

module.exports = workerProcess
