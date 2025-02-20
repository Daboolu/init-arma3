// [this] execVM 'initUAV.sqf';
waitUntil { !isNull (findDisplay 46) };
player switchCamera "External";
_entity = _this select 0;
_entity engineOn true;
_entity flyInHeight 1.5;
_entity limitSpeed 10;
_entity action ["autoHover", _entity];
_entity_dir = getDir _entity;
_entity_asl = getPosASL _entity;

_ctrl_background = findDisplay 46 ctrlCreate ["RscPicture", 1000]; 
_ctrl_background ctrlSetPosition [safeZoneX, safeZoneY, safeZoneW, safeZoneH/4 + safeZoneH*0.005];
_ctrl_background ctrlSetText "#(argb,8,8,1)color(0,0,0,0.5)"; 
_ctrl_background ctrlCommit 0; 

_ctrl_top = findDisplay 46 ctrlCreate ["RscPicture", 1001]; 
_ctrl_top ctrlSetPosition [safeZoneX, safeZoneY, safeZoneW/4, safeZoneH/4];
_ctrl_top ctrlSetText "#(rgb,8,8,1)r2t(aerial_view,1.0)"; 
_ctrl_top ctrlCommit 0; 
_cam_top = 'camera' camCreate [0,0,0];
_cam_top camSetFov 1.0;
_cam_top cameraEffect ["Internal","Back", "aerial_view"];


_ctrl_left = findDisplay 46 ctrlCreate ["RscPicture", -1]; 
_ctrl_left ctrlSetPosition [safeZoneX + safeZoneW/4, safeZoneY, safeZoneW/4, safeZoneH/4];
_ctrl_left ctrlSetText "#(rgb,512,512,1)r2t(binocular_left,1.778)"; 
_ctrl_left ctrlCommit 0; 
_cam_left = 'camera' camCreate [0,0,0];
_cam_left camSetFov 0.75;
_cam_left setDir _entity_dir;
_cam_left setPos (_entity modelToWorld [-0.05, 0.8, 0.8]);
_cam_left cameraEffect ["Internal","Back", "binocular_left"];
_cam_left camCommit 0;

_ctrl_right = findDisplay 46 ctrlCreate ["RscPicture", -1]; 
_ctrl_right ctrlSetPosition [safeZoneX + safeZoneW/2, safeZoneY, safeZoneW/4, safeZoneH/4];
_ctrl_right ctrlSetText "#(rgb,512,512,1)r2t(binocular_right,1.778)"; 
_ctrl_right ctrlCommit 0; 
_cam_right = 'camera' camCreate [0,0,0];
_cam_right camSetFov 0.75;
_cam_right setDir _entity_dir;
_cam_right setPos (_entity modelToWorld [0.05, 0.8, 0.8]);
_cam_right cameraEffect ["Internal","Back", "binocular_right"];
_cam_right camCommit 0;

[_entity, _cam_top] spawn {
    params ["_entity", "_cam_top"];
    while {true} do{
        _entity_pos = getPosASL _entity;
        _yaw = 0;
        _pitch = -90;
        _roll = 0;
        _cam_top setVectorDirAndUp [
        [sin _yaw * cos _pitch, cos _yaw * cos _pitch, sin _pitch],
        [[sin _roll, -sin _pitch, cos _roll * cos _pitch], -_yaw] call BIS_fnc_rotateVector2D];
        _cam_top setPosATL [_entity_pos select 0, _entity_pos select 1, 200];
        sleep(0.2);
    }
};

[_entity, _cam_left, _cam_right] spawn {
    params ["_entity", "_cam_left", "_cam_right"]; 
    ["agent.points_window", []] call py3_fnc_callExtension;
    sleep(1);
    ["agent.disparity_window", []] call py3_fnc_callExtension;
    _game_time = time;
    while {true} do{
        if (time - _game_time > 0) then{
            _game_time = time;
            _message = ["agent.read_message", []] call py3_fnc_callExtension;
            if (_message == "DONE") then{
                _entity_dir = getDir _entity;
                _entity_pos = getPosASL _entity;
                _cam_left setDir _entity_dir;
                _cam_left setPos (_entity modelToWorld [-0.05, 0.8, 0.8]);
                _cam_left camCommit 0;
                _cam_right setDir _entity_dir;
                _cam_right setPos (_entity modelToWorld [0.05, 0.8 , 0.8]);
                _cam_right camCommit 0;
                _message = format ["Disparity;%1,%2,%3,%4", _entity_pos select 0, _entity_pos select 1, _entity_pos select 2, _entity_dir]; 
                ["agent.send_message", [_message]] call py3_fnc_callExtension;
            };
        };        
    };
};

// _agent = uav;     
// _agent limitSpeed 5;   
// _grp = group _agent;   
// _wpPos = [4139.414, 11717.167, 1.5];     //加路径点
// _wp = _grp addWaypoint [_wpPos, -1];   
// _wp setWaypointCompletionRadius 2;  
// _wp setWaypointType "MOVE";  
// _agent flyInHeight (_wpPos select 2);  
// getPosATL _agent;
// uav1 [

_agent = uav;  
_agent flyInHeight (1.5);
_agent limitSpeed 5;  

// _agent = uav;     
// _agent limitSpeed 5;   
// _grp = group _agent;   
// _wpPos = [[4139.414, 11717.167, 1.5]];     //加路径点
// {
//     // 对每个路径点进行操作
//     _wpPos = _x;
//     _wp = _grp addWaypoint [_wpPos, -1];
//     _wp setWaypointCompletionRadius 2;
//     _wp setWaypointType "MOVE";
//     _agent flyInHeight (_wpPos select 2);
// } forEach _wpPositions;

// // 获取无人机当前位置
// _pos = getPosATL _agent;