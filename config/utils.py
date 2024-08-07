# 

import os
import streamlit as st
from temp.OneDC_Updater.update import perform_update

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        st.text('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            st.text('{}{}'.format(subindent, f))

# Streamlit app
st.title("项目文件结构")
project_dir = "C:\\Users\\xiaoy\\Downloads\\alumni-network"
list_files(project_dir)

# Update section
st.header("更新日志")
if st.button("执行更新"):
    perform_update()
    st.success("更新已执行")

log_file = os.path.join(project_dir, 'temp', 'OneDC_Updater', 'update.log')
with open(log_file, 'r') as file:
    log_contents = file.read()
    st.text(log_contents)
