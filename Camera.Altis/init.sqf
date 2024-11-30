// 定义无人机数组
_uavs = [uav1, uav2, uav3, uav4, uav5, uav6]; // 替换为你无人机实体的实际名称
_screenWidth = safeZoneW;         // 屏幕宽度
_screenHeight = safeZoneH / 3;    // 屏幕顶部区域高度（1/4屏幕高）


// 初始化每个无人机的摄像头和显示框
{
    private _uav = _x;                // 当前无人机
    private _index = _forEachIndex;   // 当前索引（0 到 3）

    // 启动无人机基本功能
    _uav engineOn true;
    _uav action ["autoHover", _uav]; // 开启悬停模式
    _uav flyInHeight 10;            // 飞行高度
    _uav limitSpeed 100;              // 最大速度
	
    // 计算框的坐标和大小
    private _boxWidth = _screenWidth / 3; // 每个框占屏幕的 1/4 宽度
    private _xPos = safeZoneX + _boxWidth * _index; // 横坐标
    private _yPos = safeZoneY + _screenHeight; // 固定在顶部
	// 两行 
	if(_index >= 3) then {
		_xPos = safeZoneX + _boxWidth * (_index-3);
		_yPos = safeZoneY + _screenHeight*2;
	};

    // 创建一个 RscPicture 控件用于显示画面
    private _ctrl = findDisplay 46 ctrlCreate ["RscPicture", 1000 + _index];
    _ctrl ctrlSetPosition [_xPos, _yPos, _boxWidth, _screenHeight]; // 设置位置和大小
    _ctrl ctrlSetText format ["#(rgb,512,512,1)r2t(uav_cam%1,1.0)", _index]; // 绑定 Render-to-Texture 名字
    _ctrl ctrlCommit 0;

    // 创建一个摄像机并绑定到无人机
    private _cam = "camera" camCreate [0, 0, 0];
    _cam camSetFov 0.75; // 设置视野范围
    _cam cameraEffect ["Internal", "Back", format ["uav_cam%1", _index]]; // 绑定到 Render-to-Texture

    
	[_cam, _uav] spawn {
    	params ["_cam", "_uav"];
    	while {true} do {
    	    // 设置摄像机位置：无人机前方 0.3 米，保持在无人机平面高度
    	    _cam setPos (_uav modelToWorld [0, 0.3, 0]);
	
     	   // 获取无人机的方向和上向量
    	    private _vectorDir = vectorDirVisual _uav; // 无人机的方向向量
    	    private _vectorUp = vectorUpVisual _uav;   // 无人机的上向量

    	    // 调整摄像机的倾斜角度，例如向下倾斜 15 度
    	    private _pitch = 15 * (pi / 180); // 转换为弧度
    	    private _adjustedDir = [
    	        (_vectorDir select 0),
    	        (_vectorDir select 1),
    	        (_vectorDir select 2) - sin(_pitch) // 调整 Z 分量，增加俯视效果
    	    ];
        // 设置摄像机方向和上向量
    	    _cam setVectorDirAndUp [_adjustedDir, _vectorUp];
        // 应用设置
    	    _cam camCommit 0;
    	    sleep 0.005; // 每 5 毫秒更新一次
    };
};


} forEach _uavs;
