require('dotenv').config();
const { notarize } = require('@electron/notarize');

exports.default = async function notarizing(context) {
  const { electronPlatformName, appOutDir } = context;
  if (electronPlatformName !== 'darwin') {
    return;
  }

  const appName = context.packager.appInfo.productFilename;
  const appBundleId = 'codecompanion';
  const appPath = `${appOutDir}/${appName}.app`;

  console.log(`Notarizing ${appBundleId} found at ${appPath}`);

  try {
    await notarize({
      tool: 'notarytool',
      appBundleId: appBundleId,
      appPath: appPath,
      appleId: process.env.APPLE_ID,
      appleIdPassword: process.env.APPLE_ID_PASSWORD,
      ascProvider: process.env.ASC_PROVIDER,
      teamId: process.env.TEAM_ID,
    });
  } catch (error) {
    console.error(error);
    throw error;
  }

  console.log(`Done notarizing ${appBundleId}`);
};
