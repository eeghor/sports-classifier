from flask import Flask
from flask import request, render_template, redirect
from flask_wtf import Form
from .forms import ContactForm

app = Flask(__name__)

@app.route("/testing", methods=['GET','POST'])
def show_form():
	
	form = ContactForm()

	if request.method == 'POST':
		return render_template('testing.html', form = form)

if __name__== "__main__":
	app.run(debug=True)


