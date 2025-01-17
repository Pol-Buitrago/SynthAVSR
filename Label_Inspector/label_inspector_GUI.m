function label_inspector_GUI
    % Initialize the GUI
    fig = uifigure('Name', 'Label-Inspector', 'Position', [300, 100, 800, 700]);

    % Create main layout with GridLayout
    layout = uigridlayout(fig, [5, 1], 'RowHeight', {70, 100, 100, 150, '1x'}, ...
                          'ColumnWidth', {'1x'});

    % Title label
    lbl_title = uilabel(layout, 'Text', 'Label-Inspector: Labeling and Video Adjustment GUI', ...
        'FontSize', 16, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

    % Labels to show file name and progress
    file_info_panel = uipanel(layout, 'Title', 'File Information', ...
        'FontSize', 12);
    file_layout = uigridlayout(file_info_panel, [1, 3]);
    lbl_file = uilabel(file_layout, 'Text', 'File: ', ...
        'FontSize', 12, 'HorizontalAlignment', 'left');
    lbl_progress = uilabel(file_layout, 'Text', 'Processed: 0 of 0', ...
        'FontSize', 12, 'HorizontalAlignment', 'right');
    btn_next = uibutton(file_layout, 'Text', 'Next', ...
        'BackgroundColor', [0.8, 0.7, 1], 'ButtonPushedFcn', @(btn, event) nextVideo());

    % Control buttons and fields
    controls_panel = uipanel(layout, 'Title', 'Controls', 'FontSize', 12);
    controls_layout = uigridlayout(controls_panel, [2, 4], ...
                                   'RowHeight', {'1x', '1x'}, ...
                                   'ColumnWidth', {'1x', '1x', '1x', '1x'});

    % Buttons and labels
    btn_play = uibutton(controls_layout, 'Text', 'Play', ...
        'BackgroundColor', [0.5, 1, 0.5], 'ButtonPushedFcn', @(btn, event) playAudio());
    btn_pause = uibutton(controls_layout, 'Text', 'Pause', ...
        'BackgroundColor', [1, 1, 0.5], 'ButtonPushedFcn', @(btn, event) pauseAudio());
    btn_cut = uibutton(controls_layout, 'Text', 'Cut', ...
        'BackgroundColor', [0.5, 0.5, 1], 'ButtonPushedFcn', @(btn, event) cutVideo());
    btn_skip = uibutton(controls_layout, 'Text', 'Invalidate', ...
        'BackgroundColor', [1, 0.5, 0.5], 'ButtonPushedFcn', @(btn, event) skipVideo());
    
    lbl_timer = uilabel(controls_layout, 'Text', 'Time: 0.0 s', ...
        'HorizontalAlignment', 'center');
    lbl_interval = uilabel(controls_layout, 'Text', 'Cut Interval (s.ms):', ...
        'HorizontalAlignment', 'center');
    edit_start = uieditfield(controls_layout, 'text', 'Placeholder', 'Start');
    edit_end = uieditfield(controls_layout, 'text', 'Placeholder', 'End');

    % Transcription TextArea
    transcription_panel = uipanel(layout, 'Title', 'Transcription', 'FontSize', 12);
    transcription_layout = uigridlayout(transcription_panel, [1, 1]);
    
    % Create a text area inside the transcription panel
    txt_transcription = uitextarea(transcription_layout, 'Value', '', 'Editable', 'on');
    txt_transcription.Layout.Row = 1;
    txt_transcription.Layout.Column = 1;
    
    % Axes for reference frames
    frames_panel = uipanel(layout, 'Title', 'Reference Frames', 'FontSize', 12);
    frames_layout = uigridlayout(frames_panel, [1, 3], ...
                                 'ColumnWidth', {'1x', '1x', '1x'});

    frame_axes1 = uiaxes(frames_layout);
    frame_axes2 = uiaxes(frames_layout);
    frame_axes3 = uiaxes(frames_layout);
    title(frame_axes1, 'First Frame');
    title(frame_axes2, 'Middle Frame');
    title(frame_axes3, 'Last Frame');
    axis(frame_axes1, 'off');
    axis(frame_axes2, 'off');
    axis(frame_axes3, 'off');

    % Variables and logic (same as before)
    validos_folder = fullfile(pwd, 'pending_review');
    revisados_folder = fullfile(pwd, 'reviewed_videos');
    invalidos_folder = fullfile(pwd, 'unvalid_videos');
    transcripciones_folder = fullfile(pwd, 'transcriptions');
    transcripciones_revisadas_folder = fullfile(pwd, 'revised_transcriptions');
    
    if ~exist(revisados_folder, 'dir'), mkdir(revisados_folder); end
    if ~exist(invalidos_folder, 'dir'), mkdir(invalidos_folder); end
    if ~exist(transcripciones_revisadas_folder, 'dir'), mkdir(transcripciones_revisadas_folder); end
    
    % List of video files that haven't been processed
    video_files = dir(fullfile(validos_folder, '*.mp4'));

    % Filter out '._' files
    video_files = video_files(~startsWith({video_files.name}, '._'));
    
    % Filter out videos that already have revised transcriptions
    video_files = video_files(~arrayfun(@(f) exist(fullfile(transcripciones_revisadas_folder, ...
        [f.name(1:end-4), '.txt']), 'file'), video_files));
    
    total_videos = numel(video_files);
    current_index = 1;
    player = [];
    
    % Show initial file info
    updateFileInfo();
    displayFrames();
    loadTranscription();
    
    % Functions (same as before)
    function playAudio()
        if current_index > total_videos
            uialert(fig, 'No more videos.', 'End');
            return;
        end
        video_file = fullfile(validos_folder, video_files(current_index).name);
        [y, Fs] = audioread(video_file); % Extract audio
        
        % Check if start and end times are provided, else play the entire audio
        start_time = str2double(edit_start.Value);
        end_time = str2double(edit_end.Value);
        
        % If start or end time is invalid or empty, play the whole audio
        if (isempty(edit_start.Value) || isempty(edit_end.Value) || ...
                isnan(start_time) || isnan(end_time) || start_time >= end_time)
            % Play entire audio if no valid start and end times are given
            start_time = 0;
            end_time = length(y) / Fs;
        end
        
        % Ensure end time doesn't exceed the audio length
        if end_time > length(y) / Fs
            end_time = length(y) / Fs;
        end
        
        % Trim audio to specified interval
        start_idx = round(start_time * Fs);
        end_idx = round(end_time * Fs);
        
        % Ensure indices are valid
        if start_idx < 1
            start_idx = 1;
        end
        if end_idx > length(y)
            end_idx = length(y);
        end
        y = y(start_idx:end_idx);
        
        player = audioplayer(y, Fs);
        player.TimerFcn = @(~, ~) updateTimer();
        player.TimerPeriod = 0.1;
        play(player);
    end

    function pauseAudio()
        if ~isempty(player)
            pause(player);
        end
    end

    function updateTimer()
        if ~isempty(player)
            lbl_timer.Text = sprintf('Time: %.1f s', player.CurrentSample / player.SampleRate);
        end
    end

    function cutVideo()
        if current_index > total_videos
            uialert(fig, 'No more videos.', 'End');
            return;
        end
        start_time = str2double(edit_start.Value);
        end_time = str2double(edit_end.Value);
        if isnan(start_time) || isnan(end_time) || start_time >= end_time
            uialert(fig, 'Invalid interval.', 'Error');
            return;
        end
        video_file = fullfile(validos_folder, video_files(current_index).name);
        output_file = fullfile(revisados_folder, video_files(current_index).name);
        recortarVideo(video_file, output_file, start_time, end_time);
        
        % Save transcription to revisadas
        transcription_text = txt_transcription.Value;
        saveTranscription(transcription_text);
        
        nextVideo();
    end

    function skipVideo()
        if current_index > total_videos
            uialert(fig, 'No more videos.', 'End');
            return;
        end
        video_file = fullfile(validos_folder, video_files(current_index).name);
        movefile(video_file, invalidos_folder);
        nextVideo();
    end

    function nextVideo()
        current_index = current_index + 1;
        if current_index > total_videos
            uialert(fig, 'No more videos.', 'End');
        else
            lbl_timer.Text = 'Time: 0.0 s';
            edit_start.Value = '';
            edit_end.Value = '';
            updateFileInfo();
            displayFrames();
            loadTranscription();
        end
    end

    function updateFileInfo()
        if current_index <= total_videos
            lbl_file.Text = sprintf('File: %s', video_files(current_index).name);
            lbl_progress.Text = sprintf('Processed: %d of %d', current_index, total_videos);
        else
            lbl_file.Text = 'No more files.';
            lbl_progress.Text = sprintf('Processed: %d of %d', total_videos, total_videos);
        end
    end

    function displayFrames()
        if current_index <= total_videos
            video_file = fullfile(validos_folder, video_files(current_index).name);
            v = VideoReader(video_file);
            frame1 = readFrame(v); % First frame
            mid_frame = round(v.Duration / 2 * v.FrameRate);
            end_frame = v.NumFrames;
            frame2 = read(v, mid_frame); % Middle frame
            frame3 = read(v, end_frame); % Last frame
            
            % Show frames on axes
            imshow(frame1, 'Parent', frame_axes1);
            imshow(frame2, 'Parent', frame_axes2);
            imshow(frame3, 'Parent', frame_axes3);
        end
    end

    function loadTranscription()
        transcription_file = fullfile(transcripciones_folder, ...
                                      [video_files(current_index).name(1:end-4), '.txt']);
        if exist(transcription_file, 'file')
            fid = fopen(transcription_file, 'r');
            transcription_text = fread(fid, '*char')';
            fclose(fid);
            txt_transcription.Value = transcription_text;
        end
    end

    function recortarVideo(input_file, output_file, start_time, end_time)
        setenv('PATH', [getenv('PATH') ':/opt/homebrew/bin']);
        command = sprintf('ffmpeg -i "%s" -ss %.2f -to %.2f -c:v libx264 -c:a aac "%s" > /dev/null 2>&1', ...
        input_file, start_time, end_time, output_file);
        system(command);
    end

    function saveTranscription(transcription_text)
        transcription_file = fullfile(transcripciones_revisadas_folder, ...
            [video_files(current_index).name(1:end-4), '.txt']);
        fid = fopen(transcription_file, 'w');

        if iscell(transcription_text)
            transcription_text = strjoin(transcription_text, '\n');
        end

        fwrite(fid, transcription_text);
        fclose(fid);
    end
end