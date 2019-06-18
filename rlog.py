# -*- coding: UTF-8 -*-

import time

LOG_ERROR_RED = '\033[41m'
LOG_WARNING_YELLOW = '\033[33m'
LOG_INFO_BLUE = '\033[34m'
LOG_NORMAL_WHITE = '\033[37m'
LOG_TOO_MUCH = '\033[31m'
LOG_BLACK = '\033[30m'
LOG_BG_BLUE = '\033[44m'
LOG_BG_PP = '\033[45m'
LOG_BG_BLACK_FG_YELLOW = '\033[33m'
LOG_BG_BLACK_FG_CYAN = '\033[36m'
LOG_END = '\033[0m'

def create_time_tag():
    time_str = time.ctime()
    time_str = time_str[4: (len(time_str) - 5)]
    time_str = '[' + time_str + '] '
    return time_str

def _log_toomuch(info, endline=False):
    print(LOG_TOO_MUCH + create_time_tag() + str(info) + LOG_END)
    if (endline):
        print('\n')

def _log_normal(info, endline=False):
    print(LOG_NORMAL_WHITE + create_time_tag() + str(info) + LOG_END)
    if (endline):
        print('\n')

def _log_warning(info, endline=False):
    print(LOG_WARNING_YELLOW + create_time_tag() + str(info) + LOG_END)
    if (endline):
        print('\n')

def _log_info(info, endline=False):
    print(LOG_INFO_BLUE + create_time_tag() + str(info) + LOG_END)
    if (endline):
        print('\n')

def _log_error(info, endline=False):
    print(LOG_ERROR_RED + create_time_tag() + str(info) + LOG_END)
    if (endline):
        print('\n')

def _log_bg_blue(info, endline=False):
    print(LOG_BG_BLUE + create_time_tag() + str(info) + LOG_END)
    if (endline):
        print('\n')

def _log_bg_pp(info, endline=False):
    print(LOG_BG_PP + create_time_tag() + str(info) + LOG_END)
    if (endline):
        print('\n')

def _log_fg_yl(info, endline=False):
    print(LOG_BG_BLACK_FG_YELLOW + create_time_tag() + str(info) + LOG_END)
    if (endline):
        print('\n')

def _log_fg_cy(info, endline=False):
    print(LOG_BG_BLACK_FG_CYAN + create_time_tag() + str(info) + LOG_END)
    if (endline):
        print('\n')

def _log_black(info, endline=False):
    print(LOG_BLACK + create_time_tag() + str(info) + LOG_END)
    if (endline):
        print('\n')