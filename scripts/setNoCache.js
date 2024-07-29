const AWS = require('aws-sdk');

const s3 = new AWS.S3({
  region: 'us-west-2',
});

const params = {
  Bucket: 'codecompanion',
  Key: '',
  Metadata: {
    'Cache-Control': 'no-cache, no-store, must-revalidate',
    Expires: '0',
  },
};

const files = ['latest-mac.yml', 'latest.yml'];

files.forEach((file) => {
  params.Key = file;
  s3.getObject(
    { Bucket: params.Bucket, Key: params.Key },
    (err, data) => {
      if (err) console.log(err, err.stack);
      else {
        params.Body = data.Body;
        s3.putObject(params, (err, data) => {
          if (err) console.log(err, err.stack);
          else console.log(data);
        });
      }
    },
  );
});
