from flask import Flask
from flask import request, render_template

app = Flask(__name__)

@app.route("/testing", methods=['GET','POST'])
def show_form():
	
	form = LoginForm()

	if request.method == 'POST':
		return render_template('testing.html', form = form)

if __name__== "__main__":
	app.run(debug=True)


