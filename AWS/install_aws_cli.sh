
# Install aws-cli if not installed
if !(type aws)
then
  # Download aws-cli
  curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
  # Unzip the downloaded folder
  unzip awscliv2.zip
  # Install aws-cli
  sudo ./aws/install
  # Remove the zipped folder
  rm awscliv2.zip
fi

# Create in home directory '.aws' for storing credentials
mkdir ~/.aws
# Move the credentials to '~/.aws/'
cp credentials ~/.aws/credentials