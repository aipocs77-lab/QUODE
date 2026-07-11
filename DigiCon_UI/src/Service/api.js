// const Ip = "192.168.43.92";
// const AnsibleIP = "192.168.43.92";
const Ip = "44.236.95.31";
const AnsibleIP = "52.33.222.223";
const baseurl = `http://${Ip}:5000/`;
const baseurlVideo = `http://${Ip}:5000`;
const apiUrl = {
  baseurl: baseurl,
  baseurlVideo: baseurlVideo,
  ansible: `http://${AnsibleIP}:8090/`,
  chatApi: baseurl + "chat",
  videoApi: baseurl + "video",
};
export default apiUrl;
