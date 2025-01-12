from flask import Flask, render_template, request
app=Flask(__name__)
import pickle
file=open('model.pkl', 'rb')

clf=pickle.load(file)
file.close()

@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method=="POST":
        myDict=request.form
        fever=int(myDict['fever'])
        bodypain=int(myDict['bodypain'])
        age=int(myDict['age'])
        bodypain=int(myDict['bodypain'])
        
        runnyNose=int(myDict['runnyNose'])
        diffBreath=int(myDict['diffBreath'])


        inputFeature=[fever, bodypain, age, runnyNose, diffBreath]
        infPro=clf.predict_proba([inputFeature])[0][1]
        print(infPro)
        return render_template('show.html', inf=round(infPro*100))
    #return 'Hello, world!'+ str(infPro)
    return render_template('index.html')

if __name__=="__main__":

    app.run(debug=True)

