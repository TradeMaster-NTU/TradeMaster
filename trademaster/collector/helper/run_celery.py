from celery_app import app as celery_app

if __name__ == '__main__':
    argv = ['worker', '--beat', '--loglevel=info', '-c', '1']
    celery_app.start(argv)