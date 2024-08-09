# !/bin/bash

# 函数：显示使用说明
usage() {
    echo "Usage: $0 command [options]"
    echo "Commands:"
    echo "  convert_repos PARENT_REPO_PATH EXTERNAL_REPOS_DIR  Convert external repos to submodules or subtrees."
    echo "  logout_users                                      Log off all users."
    exit 1
}

# 检查是否提供了命令
if [ "$#" -lt 1 ]; then
    usage
fi

# 提取命令
command=$1
shift

# 函数：转换外部仓库为子模块或子树
convert_repos() {
    if [ "$#" -ne 2 ]; then
        echo "Usage: $0 convert_repos PARENT_REPO_PATH EXTERNAL_REPOS_DIR"
        exit 1
    fi

    PARENT_REPO_PATH=$1
    EXTERNAL_REPOS_DIR=$2

    cd "$PARENT_REPO_PATH" || exit

    for repo_dir in "$EXTERNAL_REPOS_DIR"/*; do
        if [ -d "$repo_dir/.git" ]; then
            repo_url=$(git -C "$repo_dir" config --get remote.origin.url)
            if [ -z "$repo_url" ]; then
                echo "ERROR: Unable to find remote url for $repo_dir"
                continue
            fi

            repo_path="${repo_dir#$PARENT_REPO_PATH/}"

            git rm -rf "$repo_path" && rm -rf "$repo_path"
            
            # 如果你想使用子树，请取消注释以下两行，并注释掉子模块相关的行
            # git subtree add --prefix="$repo_path" "$repo_url" master --squash
            # git commit -m "Added $repo_url as a new subtree at $repo_path."

            # 如果你想使用子模块，请确保以下两行没有被注释
            git submodule add "$repo_url" "$repo_path"
        else
            echo "WARNING: $repo_dir is not a git repository."
        fi
    done

    git commit -m "Transformed external repos to submodules or subtrees."
    git submodule update --init --recursive
}

# 函数：注销所有用户
logout_users() {
    if [ "$(id -u)" != "0" ]; then
       echo "This script must be run as root" 1>&2
       exit 1
    fi

    users=$(cut -d: -f1 /etc/passwd)

    for user in $users; do
        pkill -KILL -u $user
    done

    echo "All users have been logged off."
}

# 根据命令执行相应的函数
case $command in
    convert_repos)
        convert_repos "$@"
        ;;
    logout_users)
        logout_users
        ;;
    *)
        usage
        ;;
esac
