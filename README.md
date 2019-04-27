# Client-Server
The client folder contains the html file which conatins the D3 code.
The server folder contains the data and the Roc python code in it.

Run the flask_roc.py file using the command prompt from the server folder.
From the client folder run the command 'python -m http.server' to oset up the client. 
All server requests are routed to 'http://localhost:5000/'

The client and server are expected to run seperately.

The basic HTML page gives a drop down option to select 'Max-Min' or 'Standarization'. The 'c' value(float) needs to be entered in a text box.
The 'generate roc' button will create a Roc curve according to the parameters pased.
