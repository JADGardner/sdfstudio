FROM dromni/nerfstudio:0.1.14

USER root

RUN apt-get update && \
    apt-get install curl 
    
RUN apt-get install openssh-server -y supervisor   

# make .ssh directory and set permissions for user
RUN mkdir -p /home/user/.ssh && \
    chown -R user:user /home/user/.ssh && \
    chmod 700 /home/user/.ssh

USER 1000:1000

# update pip 
RUN pip install -U pylint
RUN pip install -U black
RUN pip install -U yacs
RUN pip install -U icosphere