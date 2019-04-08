from subprocess import call
import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
import constants.databases_info as c

def take_pic():
    #raspistill -o image1.jpg
    
    now = datetime.datetime.now()
    formatted_now = now.strftime('%d-%m-%Y_%H-%M-%S')
    print("Taking picture {}".format(formatted_now))
    call(["raspistill", "-o", "/home/pi/parking_python/image{}.jpg".format(formatted_now)])

scheduler = BlockingScheduler()
scheduler.add_job(take_pic, 'interval', seconds=c.UACJ_picture_lapse)
scheduler.start()


