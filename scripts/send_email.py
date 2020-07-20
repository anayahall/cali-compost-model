# email.py

import pickle
import yagmail

yag = yagmail.SMTP('anayahall@gmail.com', 'zpnzmthmdosrdcff')
contents = [
    "This is the body, and here is just text",
    "You can find an audio file attached."
]

filename = "./imagetest_c2f.p"

yag.send(
    to='anayahall@berkeley.edu',
    subject="Yagmail test with attachment",
    contents='body', 
    attachments=filename,
)

