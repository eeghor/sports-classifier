from flask import Flask
from flask import request, render_template, redirect
from flask_wtf import Form
from wtforms import StringField
from wtforms.validators import DataRequired
from flask.ext.bootstrap import Bootstrap

app = Flask(__name__)
app.config.from_object('config')
bootstrap = Bootstrap(app)

# make a new class (which is effectively the form) based on the base class Form
class SportStringForm(Form):
	descr_str = StringField(u'event description', validators=[DataRequired()])
	res_str = StringField(u'result', validators=[DataRequired()])

@app.route("/testing", methods=['GET','POST'])
def show_form():
	
	form = SportStringForm()

	if request.method == "POST":
		st = request.form["descr_str"]
		print(st)
	return render_template('sports_text.html', form = form)


if __name__== "__main__":
	app.run(debug=True)


