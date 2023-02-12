from flask import Flask

app = Flask(__name__) 
app.config['SECRET_KEY'] = 'random_secret_key'

######################################## FORMS
from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField
from wtforms.validators import DataRequired

class RandomForm(FlaskForm):
    
    text = TextAreaField(label='Type Your Sentence', validators=[DataRequired()])
    submit = SubmitField('Enter')

######################################## VIEWS
from flask import render_template, redirect, url_for, request

@app.route('/', methods=['GET','POST'])
def index():
    form = RandomForm()
    if request.method == 'POST':
    # if form.validate_on_submit():
    #     print(form.text.data)
    #     return redirect(url_for('index', form=form))
    return render_template('index.html', form=form)


if __name__ == '__main__':
    app.debug = True
    app.run(host = '0.0.0.0', port = 5001)