from flask import Flask,flash, request, redirect, url_for
from flask import render_template
from werkzeug.utils import redirect
import os
from werkzeug.utils import secure_filename

import settings
app = Flask(__name__)
app.config.from_object(settings)

data = {'a':'北京','b':'上海','c':'深圳'}

@app.route('/index')
def index():
    return '我是你爸爸'

@app.route('/get_city/<key>')
def get_city(key):
    return data.get(key)

@app.route('/register',methods=['GET','POST'])
def register():
    r = render_template('flask-1.html') #可以返回HTML文件.将HTML转换成。jinjia2是模板引擎，它会自动去templates文件夹中找到你所写的HTML文件，在转成字符串
    return r

@app.route('/register2',methods=['GET','POST']) #允许GET和POST请求的方式
def register2():
    print(request.args)

    # 获取GET请求的表单中提交的元素，name所对应的属性.
    username = request.args.get('username')
    password = request.args.get('password')

    # 获取POST请求的表单中提交的元素，name所对应的属性.
    username = request.form.get('username')
    password = request.form.get('password')

    print(username,password)
    return '进来了'

@app.route('/register3',methods=['GET','POST']) #允许GET和POST请求的方式
def register3():
    print(request.args)

    # 获取GET请求的表单中提交的元素，name所对应的属性.
    if request.method == 'GET':
        username = request.args.get('username')
        password = request.args.get('password')

    # 获取POST请求的表单中提交的元素，name所对应的属性.
    if request.method == 'POST':
        username = str(request.form.get('username'))
        password = str(request.form.get('password'))
        print(username, password)

    if username == '18339189932':
        if password == '123456':
           # r = render_template('flask-1.html')
           # return r
            return redirect('http://117.25.169.110:524/')

    return '密码不正确'

#@app.route('/uplode_file',methods=['GET','POST'])
#def uplode_file():
    #UPLOAD_FOLDER = '/path/to/the/uploads'
    #app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
   # file = request.files['file']
    #print(file)
   # filename = secure_filename(file.filename)
    #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(host = '0.0.0.0',port = 8080)
#1111111111