#add base image
FROM python:3.11-slim

# add working directory
WORKDIR /app

#copy all file
copy . /app
#install all libraries
run pip install -r requirements.txt
#export port
EXPOSE 8080
#command to run app
CMD ["python", "01_build_master_index.py"]
