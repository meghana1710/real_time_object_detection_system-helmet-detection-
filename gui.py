import sqlite3
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime
import os
from tkinter import * 
from tkinter.ttk import *
from reportlab.lib.pagesizes import letter
import datetime
import smtplib
from pdf_mail import sendpdf

master = Tk() 
  
master.geometry("400x400") 
  
 
def openNewWindow(): 
       
    newWindow = Toplevel(master) 

    newWindow.title("Database entry") 

    newWindow.geometry("400x400") 

    l0 = Label(newWindow, text ="Database entry")
    l0.grid(row=0,column=0)
    l1 = Label(newWindow, text ="License plate")
    l1.grid(row=1,column=0)
    e1 = Entry(newWindow)
    e1.grid(row=1,column=1)
    
    l2 = Label(newWindow, text ="Name on registration")
    l2.grid(row=2,column=0)
    e2 = Entry(newWindow)
    e2.grid(row=2,column=1)

    l3 = Label(newWindow, text ="Address")
    l3.grid(row=3,column=0)
    e3 = Entry(newWindow)
    e3.grid(row=3,column=1)

    l4 = Label(newWindow, text ="Email")
    l4.grid(row=4,column=0)
    e4 = Entry(newWindow)
    e4.grid(row=4,column=1)
    
    btn4 = Button(newWindow,  
             text ="Click to enter records",  
             command = lambda : enter_records(e1,e2,e3,e4)) 
    btn4.grid(row=5,column=0)
          
          
          
def enter_records(e1,e2,e3,e4):
    conn = sqlite3.connect('vehicledata.db')

    cursor = conn.cursor()
    
    cursor.execute("insert into vehicle values('{}','{}','{}','{}')".format(e1.get(),e2.get(),e3.get(),e4.get()))
    

    conn.commit()
    print("Records added...")
    conn.close()

          
    
def run_program():

    os.system('py detect.py')    
            
def generate_challans():
    con=sqlite3.connect('vehicledata.db')

    cursor=con.cursor()
    lplate=input("Enter the number plate to generate chalan:")
    cursor.execute("select * from vehicle where plate='{}';".format(lplate))
    result=cursor.fetchone();
    #print(result)
    if result is None:
        print("No data available. Add the new record")
        #break
    else:
        name=result[1]
        address=result[2]
        email=result[3]

    con.close()

    PATH_TO_PDF ='D:/sample.pdf'
    can=canvas.Canvas(PATH_TO_PDF)
    can.setLineWidth(30)
    can.setFont('Helvetica', 12)
    
    can.drawString(30,750,'CHALLAN')
    can.drawString(30,730,'TRAFFIC POLICE DEPARTMENT, KAKINADA')
    can.drawString(30,710,str(datetime.datetime.now()))
    #can.line(10,707,580,707)

    can.drawString(30,650,'VIOLATION:')
    can.drawString(300,650,"NOT WEARING HELMET")

    can.drawString(30,600,'LICENSE PLATE:')
    can.drawString(300,600,lplate)

    can.drawString(30,550,'NAME:')
    can.drawString(300,550,name)

    can.drawString(30,500,'ADDRESS:')
    can.drawString(300,500,address)

    can.drawString(30,450,'EMAIL:')
    can.drawString(300,450,email)
        
    can.drawString(30,400,'FINE:')
    can.drawString(300,400,'Rs. 500')
      
    can.save()
    
    print("Challan generated.")
    sender_email_address = "samples123.1234@gmail.com"
    receiver_email_address = email
    sender_email_password = "8886897577"
    subject_of_email = "E-Challan for violating the traffic rules"
    body_of_email = "Fine to be paid"
    filename = "sample"
    location_of_file = 'D:/'
    k = sendpdf(sender_email_address, 
            receiver_email_address,
            sender_email_password,
            subject_of_email,
            body_of_email,
            filename,
            location_of_file)
    k.email_send()
    
    print("Mail sent")
  
  
label = Label(master,  
              text ="This is the main window") 
  
label.pack(pady = 10) 
  

btn = Button(master,  
             text ="Click to open database entry",  
             command = openNewWindow) 
btn.pack(pady = 10) 


btn2 = Button(master,  
             text ="Click to run helmet detection program",  
             command = run_program) 
btn2.pack(pady = 10) 
  
  
btn3 = Button(master,  
             text ="Click to generate challans",  
             command = generate_challans) 
btn3.pack(pady = 10) 


