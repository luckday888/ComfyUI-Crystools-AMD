import { utils } from './comfy/index.js';
// CSS 路径相对当前模块解析，避免写死插件目录名（ComfyUI-Crystools / ComfyUI-Crystools-AMD 均可用），
// 否则 monitor.css 会 404，监控条只剩文字、彩色图形条不显示。
utils.addStylesheet(new URL('./monitor.css', import.meta.url).href);
export var Styles;
(function (Styles) {
    Styles["BARS"] = "BARS";
})(Styles || (Styles = {}));
export var Colors;
(function (Colors) {
    Colors["CPU"] = "#0AA015";
    Colors["RAM"] = "#07630D";
    Colors["DISK"] = "#730F92";
    Colors["GPU"] = "#0C86F4";
    Colors["VRAM"] = "#176EC7";
    Colors["TEMP_START"] = "#00ff00";
    Colors["TEMP_END"] = "#ff0000";
})(Colors || (Colors = {}));
