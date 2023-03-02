# https://stackoverflow.com/questions/12202587/automatically-enter-ssh-password-with-script
# https://docs.paramiko.org/en/stable/api/client.html

import paramiko

ssh_client = paramiko.SSHClient()

# To avoid an "unknown hosts" error. Solve this differently if you must...
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# # This mechanism uses a private key.
# pkey = paramiko.RSAKey.from_private_key_file(PKEY_PATH)

# This mechanism uses a password.
# Get it from cli args or a file or hard code it, whatever works best for you
password = "**********"

for ip in range(11, 80):
    hostname_ = "10.127.30." + str(ip)
    try:
        ssh_client.connect(hostname=hostname_,
                            username="umar.salman",
                            password=password
                            # pkey=pkey
                            )
        command = "nvidia-smi | grep 'Default' | awk '{ print $9 \" out of \"  $11 \" - GPU Util: \" $13}'"
        stdin, stdout, stderr = ssh_client.exec_command(command)
        print(hostname_)
        for line in stdout.readlines():
            print(line)
        print('-'*20)
        ssh_client.close()
    except:
        print(hostname_ + ' NOT REACHABLE')