import { utils } from './comfy/index.js';

// CSS 路径相对当前模块解析，不要写死插件目录名：
// 目录名可能是 ComfyUI-Crystools（上游）或 ComfyUI-Crystools-AMD（本分支），
// 写死会导致 monitor.css 404、监控条只剩文字没有彩色图形条。
utils.addStylesheet(new URL('./monitor.css', import.meta.url).href);

export enum Styles {
  'BARS' = 'BARS'
}

export enum Colors {
  'CPU' = '#0AA015',
  'RAM' = '#07630D',
  'DISK' = '#730F92',
  'GPU' = '#0C86F4',
  'VRAM' = '#176EC7',
  'TEMP_START' = '#00ff00',
  'TEMP_END' = '#ff0000',
}
